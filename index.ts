/**
 * Session Review Extension
 *
 * Spawns a separate analysis session that receives the full context of the
 * current session (system prompt, tool definitions, and conversation) and
 * runs a user-chosen analysis prompt against it.
 *
 * Usage:
 *   /review            - Pick an analysis preset from the configured list
 *   /review <prompt>   - Run a custom ad-hoc analysis prompt
 *
 * Configuration:
 *   Place preset prompts as .md files in the `prompts/` directory next to
 *   this extension. Each file becomes a selectable preset.
 *
 *   File format:
 *     ---
 *     name: My Preset
 *     description: What this preset analyzes
 *     ---
 *     The actual analysis prompt template goes here.
 *     Use {SESSION_CONTEXT} as placeholder for the serialized session data.
 *
 *   If no {SESSION_CONTEXT} placeholder is present, the session context is
 *   prepended automatically.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { pathToFileURL } from "node:url";

import { complete, type Message } from "@mariozechner/pi-ai";
import type { ExtensionAPI, ExtensionCommandContext } from "@mariozechner/pi-coding-agent";
import {
  BorderedLoader,
  convertToLlm,
  getMarkdownTheme,
  parseFrontmatter,
  serializeConversation,
} from "@mariozechner/pi-coding-agent";
import { Container, Markdown, matchesKey, Text } from "@mariozechner/pi-tui";

// ── Types ──────────────────────────────────────────────────────────────

interface Preset {
  name: string;
  description: string;
  prompt: string;
  filePath: string;
}

interface SessionContext {
  systemPrompt: string;
  tools: Array<{ name: string; description: string; parameters: unknown }>;
  conversation: string;
  model?: string;
}

// ── Preset discovery ───────────────────────────────────────────────────

function loadPresets(extensionDir: string): Preset[] {
  const promptsDir = path.join(extensionDir, "prompts");
  if (!fs.existsSync(promptsDir)) return [];

  const presets: Preset[] = [];

  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(promptsDir, { withFileTypes: true });
  } catch {
    return [];
  }

  for (const entry of entries) {
    if (!entry.name.endsWith(".md")) continue;
    if (!entry.isFile() && !entry.isSymbolicLink()) continue;

    const filePath = path.join(promptsDir, entry.name);
    let content: string;
    try {
      content = fs.readFileSync(filePath, "utf-8");
    } catch {
      continue;
    }

    const { frontmatter, body } = parseFrontmatter<Record<string, string>>(content);

    const name = frontmatter.name || entry.name.replace(/\.md$/, "");
    const description = frontmatter.description || "";

    presets.push({ name, description, prompt: body.trim(), filePath });
  }

  return presets.sort((a, b) => a.name.localeCompare(b.name));
}

// ── Session context extraction ─────────────────────────────────────────

function extractSessionContext(ctx: ExtensionCommandContext, pi: ExtensionAPI): SessionContext {
  const branch = ctx.sessionManager.getBranch();
  const messages = branch
    .filter(
      (entry): entry is (typeof branch)[number] & { type: "message" } => entry.type === "message",
    )
    .map((entry) => entry.message);

  const llmMessages = convertToLlm(messages);
  const conversation = serializeConversation(llmMessages);

  const tools = pi.getAllTools().map((t) => ({
    name: t.name,
    description: t.description,
    parameters: t.parameters,
  }));

  const systemPrompt = ctx.getSystemPrompt();
  const model = ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : undefined;

  return { systemPrompt, tools, conversation, model };
}

function serializeSessionContext(sc: SessionContext): string {
  const toolSection = sc.tools
    .map((t) => {
      const params = JSON.stringify(t.parameters, null, 2);
      return `### ${t.name}\n${t.description}\n\nParameters:\n\`\`\`json\n${params}\n\`\`\``;
    })
    .join("\n\n");

  return [
    "# Session Under Review",
    "",
    `**Model:** ${sc.model ?? "unknown"}`,
    "",
    "## System Prompt",
    "",
    "```",
    sc.systemPrompt,
    "```",
    "",
    "## Available Tools",
    "",
    toolSection,
    "",
    "## Conversation",
    "",
    sc.conversation,
  ].join("\n");
}

// ── Build the analysis prompt ──────────────────────────────────────────

function buildAnalysisPrompt(preset: { prompt: string }, sessionContextText: string): string {
  if (preset.prompt.includes("{SESSION_CONTEXT}")) {
    return preset.prompt.replace(/\{SESSION_CONTEXT\}/g, sessionContextText);
  }
  return `${sessionContextText}\n\n---\n\n${preset.prompt}`;
}

// ── Analysis via direct LLM call ───────────────────────────────────────

const ANALYSIS_SYSTEM_PROMPT = `You are an expert session analyst. You review coding agent sessions to help the user understand what happened, identify issues, and improve their workflows.

You will receive the full context of an agent session including:
- The system prompt the agent was given
- The tools available to the agent
- The full conversation history

Analyze the session carefully and provide a thorough, structured report based on the user's analysis request.

Create a detailed html report. Visualize issues were appropriate.`;

async function runAnalysis(
  analysisPrompt: string,
  ctx: ExtensionCommandContext,
  pi: ExtensionAPI,
  signal: AbortSignal,
): Promise<string | null> {
  if (!ctx.model) throw new Error("No model selected");

  const auth = await ctx.modelRegistry.getApiKeyAndHeaders(ctx.model);
  if (!auth.ok || !auth.apiKey) {
    throw new Error(auth.ok ? `No API key for ${ctx.model.provider}` : auth.error);
  }

  const userMessage: Message = {
    role: "user",
    content: [{ type: "text", text: analysisPrompt }],
    timestamp: Date.now(),
  };

  const thinkingLevel = pi.getThinkingLevel();
  const response = await complete(
    ctx.model,
    { systemPrompt: ANALYSIS_SYSTEM_PROMPT, messages: [userMessage] },
    {
      apiKey: auth.apiKey,
      headers: auth.headers,
      signal,
      reasoningEffort: thinkingLevel === "off" ? undefined : thinkingLevel,
    },
  );

  if (response.stopReason === "aborted") return null;

  return response.content
    .filter((c): c is { type: "text"; text: string } => c.type === "text")
    .map((c) => c.text)
    .join("\n");
}

// ── Result display for markdown ────────────────────────────────────────

async function showReport(
  report: string,
  presetName: string,
  ctx: ExtensionCommandContext,
): Promise<void> {
  if (!ctx.hasUI) return;

  await ctx.ui.custom((_tui, theme, _kb, done) => {
    const container = new Container();
    const mdTheme = getMarkdownTheme();

    const header = theme.fg("accent", theme.bold(`Session Review: ${presetName}`));
    container.addChild(new Text(header, 1, 0));
    container.addChild(new Text(theme.fg("dim", "─".repeat(60)), 1, 0));
    container.addChild(new Markdown(report, 1, 1, mdTheme));
    container.addChild(new Text(theme.fg("dim", "─".repeat(60)), 1, 0));
    container.addChild(new Text(theme.fg("dim", "Press Enter or Esc to close"), 1, 0));

    return {
      render: (width: number) => container.render(width),
      invalidate: () => container.invalidate(),
      handleInput: (data: string) => {
        if (matchesKey(data, "enter") || matchesKey(data, "escape")) {
          done(undefined);
        }
      },
    };
  });
}

// ── Share session via GitHub gist ──────────────────────────────────────

type ExportSessionToHtml = (
  sessionManager: any,
  state?: {
    systemPrompt?: string;
    tools?: Array<{ name: string; description?: string; parameters?: unknown }>;
  },
  options?: string | { outputPath?: string },
) => Promise<string>;

let _exportSessionToHtml: ExportSessionToHtml | undefined;

async function getExportFn(): Promise<ExportSessionToHtml> {
  if (_exportSessionToHtml) return _exportSessionToHtml;

  const piEntryUrl = import.meta.resolve("@mariozechner/pi-coding-agent");
  const piEntry = new URL(piEntryUrl).pathname;
  const exportModulePath = path.join(path.dirname(piEntry), "core", "export-html", "index.js");
  const mod = (await import(pathToFileURL(exportModulePath).href)) as {
    exportSessionToHtml: ExportSessionToHtml;
  };
  _exportSessionToHtml = mod.exportSessionToHtml;
  return _exportSessionToHtml;
}

function getShareViewerUrl(gistId: string): string {
  const base = process.env.PI_SHARE_VIEWER_URL || "https://pi.dev/session/";
  const normalizedBase = base.includes("#") ? base.replace(/#.*$/, "#") : `${base}#`;
  return `${normalizedBase}${gistId}`;
}

/**
 * Exports the current session as HTML, uploads it as a secret GitHub gist,
 * and returns the pi.dev/session viewer URL (or null on failure).
 *
 * Optimised: skips the separate `gh auth status` check (gist create will
 * fail fast if unauthenticated) and fires the description-edit without
 * waiting for it to finish.
 */
async function shareSessionAsGist(
  pi: ExtensionAPI,
  ctx: ExtensionCommandContext,
  name: string,
): Promise<string | null> {
  const sessionFile = ctx.sessionManager.getSessionFile();
  if (!sessionFile) return null;

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-session-review-"));
  const tmpFile = path.join(tmpDir, "session.html");

  try {
    // Export current session to HTML
    const allTools = new Map(pi.getAllTools().map((tool) => [tool.name, tool] as const));
    const activeTools = pi
      .getActiveTools()
      .map((name) => allTools.get(name))
      .filter((tool): tool is NonNullable<typeof tool> => Boolean(tool))
      .map((tool) => ({
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,
      }));

    let systemPrompt: string | undefined;
    try {
      systemPrompt = ctx.getSystemPrompt();
    } catch {
      systemPrompt = undefined;
    }

    const exportSessionToHtml = await getExportFn();
    await exportSessionToHtml(ctx.sessionManager, { systemPrompt, tools: activeTools }, tmpFile);

    // Create secret gist — no prior auth check; gh will fail fast if not logged in
    const result = await pi.exec(
      "gh",
      ["gist", "create", "--desc", `session-review: ${name}`, tmpFile],
      { signal: ctx.signal },
    );

    if (result.code !== 0) return null;

    const gistUrl = result.stdout
      .trim()
      .split(/\s+/)
      .find((part) => part.startsWith("https://gist.github.com/"));
    const gistId = gistUrl?.split("/").pop();
    if (!gistId) return null;

    const previewUrl = getShareViewerUrl(gistId);

    // Fire-and-forget: update gist description with viewer URL
    const fullDesc = `session-review: ${name} — ${previewUrl}`;
    pi.exec("gh", ["gist", "edit", gistId, "--desc", fullDesc]).catch(() => {});

    return previewUrl;
  } catch {
    return null;
  } finally {
    try {
      fs.rmSync(tmpDir, { recursive: true });
    } catch {
      // Ignore cleanup errors
    }
  }
}

/**
 * Inject a share link into the HTML report right after the first <h1> tag.
 * Falls back to prepending if no <h1> is found.
 */
function injectShareLink(html: string, shareUrl: string): string {
  const linkHtml = `<p style="margin-top:4px"><a href="${shareUrl}" target="_blank" rel="noopener">View full session transcript ↗</a></p>`;
  const h1Close = html.indexOf("</h1>");
  if (h1Close !== -1) {
    const insertAt = h1Close + "</h1>".length;
    return html.slice(0, insertAt) + "\n" + linkHtml + html.slice(insertAt);
  }
  // No <h1> found — prepend
  return linkHtml + "\n" + html;
}

// ── Extension ──────────────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  const extensionDir = path.dirname(new URL(import.meta.url).pathname);

  pi.registerCommand("review", {
    description: "Analyze the current session with a review preset or custom prompt",
    handler: async (args, ctx) => {
      if (!ctx.hasUI) {
        ctx.ui.notify("review requires interactive mode", "error");
        return;
      }

      if (!ctx.model) {
        ctx.ui.notify("No model selected", "error");
        return;
      }

      // Wait for any running agent to finish
      await ctx.waitForIdle();

      // Check if there's a conversation to review
      const branch = ctx.sessionManager.getBranch();
      const hasMessages = branch.some(
        (e) =>
          e.type === "message" && (e.message.role === "user" || e.message.role === "assistant"),
      );

      if (!hasMessages) {
        ctx.ui.notify("No conversation to review", "warning");
        return;
      }

      let selectedPreset: { name: string; prompt: string };

      if (args.trim()) {
        // Ad-hoc prompt from command argument
        selectedPreset = { name: "Custom", prompt: args.trim() };
      } else {
        // Load presets and let user pick
        const presets = loadPresets(extensionDir);

        if (presets.length === 0) {
          ctx.ui.notify(
            `No presets found. Add .md files to:\n${path.join(extensionDir, "prompts/")}`,
            "warning",
          );
          return;
        }

        const options = presets.map((p) =>
          p.description ? `${p.name} – ${p.description}` : p.name,
        );
        const choice = await ctx.ui.select("Select analysis preset:", options);

        if (!choice) return; // cancelled

        const choiceIndex = options.indexOf(choice);
        selectedPreset = presets[choiceIndex];
      }

      // Extract full session context
      const sessionContext = extractSessionContext(ctx, pi);
      const sessionContextText = serializeSessionContext(sessionContext);
      const analysisPrompt = buildAnalysisPrompt(selectedPreset, sessionContextText);

      const safeName = selectedPreset.name.replace(/[^\w.-]+/g, "_").toLowerCase();

      // Kick off gist creation and LLM analysis in parallel.
      // The LLM call dominates wall-clock time so the gist is typically
      // ready well before the report finishes.
      const sharePromise = shareSessionAsGist(pi, ctx, safeName);

      // Run analysis with loading UI
      const report = await ctx.ui.custom<string | null>((tui, theme, _kb, done) => {
        const loader = new BorderedLoader(
          tui,
          theme,
          `Analyzing session: ${selectedPreset.name}...`,
        );
        loader.onAbort = () => done(null);

        runAnalysis(analysisPrompt, ctx, pi, loader.signal)
          .then(done)
          .catch((err) => {
            console.error("Session review failed:", err);
            done(null);
          });

        return loader;
      });

      if (!report) {
        ctx.ui.notify("Review cancelled", "info");
        return;
      }

      // Await the share URL (likely already resolved) and inject into HTML
      const shareUrl = await sharePromise;
      const finalReport = shareUrl ? injectShareLink(report, shareUrl) : report;

      // Save report to the working directory (where the agent was started)
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      const outputPath = path.join(ctx.cwd, `${timestamp}_${safeName}.html`);
      fs.writeFileSync(outputPath, finalReport, "utf-8");

      ctx.ui.notify(`Report saved to ${outputPath}`, "info");
    },
  });
}
