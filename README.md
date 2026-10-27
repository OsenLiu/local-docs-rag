# Local Docs RAG

Local hybrid search and cited question answering for PDF and Markdown documents. The index stays on this Mac; embedding and answer text are sent to the configured OpenAI-compatible provider when those features are enabled.

## Install

```bash
cd ~/Projects/local-docs-rag
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

## Configure

Run `local-docs-rag setup` and provide:

- Legacy document directory used when adopting an existing index (defaults to `~/Downloads/doc`)
- SQLite index location (defaults to `~/Library/Application Support/local-docs-rag/index.sqlite3`)
- OpenAI-compatible API base URL, embedding model, embedding dimensions, and answer model

Non-secret settings are written to `~/Library/Application Support/local-docs-rag/config.json`. The API key is stored in macOS Keychain:

```bash
.venv/bin/local-docs-rag credentials set
```

To create defaults without prompts and use keyword-only search until a model is selected, run `.venv/bin/local-docs-rag setup --defaults`.

`RAG_API_KEY` can temporarily override the Keychain value. Other settings can be overridden with `RAG_SOURCE_DIR`, `RAG_DATABASE_PATH`, `RAG_BASE_URL`, `RAG_EMBEDDING_MODEL`, `RAG_EMBEDDING_DIMENSIONS`, and `RAG_ANSWER_MODEL`.

When embedding settings are absent, indexing and searching use FTS5 only. Configure both the embedding model and its exact output dimension to enable semantic retrieval. Embedding and answer model identifiers depend on the provider account; this project does not assume a provider or model.

### Projects

Register each project with a lowercase slug. A project can contain indexes from
many directories; projects share the configured SQLite database, with every
index, search, answer, and evaluation operation scoped to its explicit project:

```bash
.venv/bin/local-docs-rag project add team-docs
.venv/bin/local-docs-rag project list
```

Run `local-docs-rag index team-docs` from each directory whose documents should
belong to `team-docs`. The first index from a directory registers it. Re-indexing
from that same directory synchronizes its files, including removing indexed
files deleted there, while retaining indexes from other directories in the
project. Use `project remove-dir team-docs DIRECTORY --yes` to remove one
directory and its index; `project remove team-docs --yes` removes the entire
project and its index.

The project registry is stored in the global settings file. Embedding and answer
model settings are shared across projects. Project labels prevent accidental
cross-project mixing; they are not an access-control boundary against another
user who can access the local database. To preserve an existing unlabelled index,
use `project add NAME --adopt-existing`; the legacy configured source directory
is attached to the project. Project and directory removal require confirmation
and never delete source files.

## Use From The Terminal

```bash
# Run index from each source directory to include it in the project.
cd ~/Projects/team/docs
~/Projects/local-docs-rag/.venv/bin/local-docs-rag index team-docs

# These data commands require a project name and can run from any directory.
~/Projects/local-docs-rag/.venv/bin/local-docs-rag search team-docs "trusted network setup"
~/Projects/local-docs-rag/.venv/bin/local-docs-rag ask team-docs "How do I reset the device?"
~/Projects/local-docs-rag/.venv/bin/local-docs-rag evaluate team-docs
```

`index PROJECT` processes the current working directory, accepts only `.pdf` and
`.md`, and detects changes with SHA-256. It removes deleted files only from that
directory's index, retaining data indexed from other directories in the same
project. Each PDF page and Markdown heading is retained as a citation locator.
Empty or unreadable documents appear in the index report; the first release does
not perform OCR.

## Connect VS Code

Add this server to `.vscode/mcp.json` in a workspace:

```json
{
  "servers": {
    "local-docs-rag": {
      "type": "stdio",
      "command": "${userHome}/Projects/local-docs-rag/.venv/bin/python",
      "args": ["-m", "local_docs_rag.mcp_server"]
    }
  }
}
```

The server reads the same user settings and shared SQLite database as the
terminal commands. Each tool call must include its `project` argument; retrieval
and citations are restricted to that registered project.

### Choose the right tool

This MCP server exposes two tools:

- `search_documents(project, query, limit)` returns passages and citations from only the selected project. Copilot can use these results to write its response using your existing Copilot sign-in; this does not require a model-provider API key if semantic embeddings are not configured.
- `answer_documents(project, query, limit)` generates an answer using evidence from only the selected project and the configured answer model. It does **not** use your GitHub Copilot sign-in. It requires an API key for the configured OpenAI-compatible provider, saved with `local-docs-rag credentials set` (or supplied temporarily as `RAG_API_KEY`). If no answer model is configured, this tool returns an abstention rather than a generated answer.

VS Code or Copilot may select `answer_documents` when deciding which tool to call. If you want Copilot to answer from retrieved passages with your Copilot sign-in instead, ask it to use `search_documents` and then synthesize the results. Copilot authentication and this MCP server's model-provider credentials are separate.

You can make this preference persistent in workspace instructions. For example, add this to `.github/copilot-instructions.md` or `AGENTS.md`:

```markdown
For questions about my local documents, determine the relevant registered project
and pass it as `project` to the `search_documents` tool from the `local-docs-rag`
MCP server. If the project is unclear, ask me. Answer using only the returned
passages and citations from that project.
Do not use `answer_documents` unless I explicitly ask for an answer generated by
the configured model provider.
```

For a dedicated VS Code custom agent, create `.github/agents/Local Docs.agent.md` and limit its available tools in the frontmatter:

```markdown
---
name: Local Docs
description: Answer questions using my local document index and Copilot.
tools: ["local-docs-rag/search_documents"]
---

For local document questions, include the user's registered project in the
`project` argument to `search_documents` and base your response only on the
returned passages and citations from that project.
```

The instruction file tells Copilot which tool to prefer; the custom agent's `tools` list makes only the specified MCP tool available to that agent. See the VS Code docs for [custom instructions](https://code.visualstudio.com/docs/copilot/customization/custom-instructions) and [custom agents](https://code.visualstudio.com/docs/copilot/customization/custom-agents).

## Connect GitHub Copilot CLI

Copilot CLI has its own MCP configuration; it does not read VS Code's `.vscode/mcp.json`:

```bash
copilot mcp add local-docs-rag -- "$HOME/Projects/local-docs-rag/.venv/bin/python" -m local_docs_rag.mcp_server
```

Both clients start the same stdio server and use the shared local database and
Keychain credential. The server serves multiple registered projects; each tool
call must supply `project`, and its results are restricted to that project. In
Copilot CLI, `/mcp` shows the registered tools.

### Install the Copilot plugin

The plugin searches the local index by invoking the `local-docs-rag` CLI directly; it does not start or configure an MCP server. Install the CLI from this checkout so it is available in every terminal:

```bash
pipx install --editable "$PWD"
pipx ensurepath
```

Restart your terminal if `pipx ensurepath` changed `PATH`. After the plugin changes have been pushed to GitHub, install the plugin from the repository:

```bash
copilot plugin install OsenLiu/local-docs-rag
```

The plugin is installed for future Copilot CLI sessions. Confirm it is listed with `copilot plugin list`; then start a new session and ask about your local documents or invoke `/local-docs-rag`. The skill asks for or determines the relevant project, runs `local-docs-rag search PROJECT QUERY`, and asks Copilot to answer from that project's returned passages with source paths and locators. Update the plugin with `copilot plugin update local-docs-rag`.

To try the plugin from an unpublished local checkout instead, run `copilot --plugin-dir "$PWD"` from the repository root. This loads it for that session only. The manual MCP setup above remains available separately for MCP clients.

## Data Handling

- Original files are read in place and are not copied into the index directory.
- The shared SQLite database stores extracted chunks, source paths, locators, hashes, vectors, and project labels locally; search and index maintenance are scoped to one project.
- With semantic retrieval enabled, only the selected project's document chunks are sent to the embedding provider during indexing and questions are embedded during search.
- For generated answers, the query and evidence retrieved from the selected project are sent to the answer provider.
- API keys are not written to the settings file or MCP configuration.
- No OCR, reranker, or ingestion of patches, images, or archives is enabled in the first release.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The real-document evaluation questions and results are git-ignored because they reveal private document topics. The local 22-question set stores queries, expected source paths, and page/title locators only; it does not duplicate document text. Run `local-docs-rag evaluate PROJECT` after indexing to evaluate only that project's index. By default, evaluation reads `tests/evaluation/questions.jsonl` and writes metrics to `tests/evaluation/results-PROJECT.json`. Specify alternate files with `--questions-path` and `--output-path`:

```bash
local-docs-rag evaluate team-docs \
  --questions-path /path/to/questions.jsonl \
  --output-path /path/to/results.json
```

Metrics include Recall@5, locator accuracy, and Chinese-query Recall@5. These are baselines, not CI thresholds. Inspect answer citations manually for evidence support; the automated tests only reject citation IDs outside retrieved evidence.