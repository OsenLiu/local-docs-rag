---
name: local-docs-rag
description: 'Search and answer questions from the local PDF and Markdown document index. Use when a user asks about their local documents or requests evidence from indexed files.'
---

# Local document search

Use the `local-docs-rag` CLI to retrieve evidence, then answer using Copilot.
Select the registered project that matches the user's question. If it is unclear,
ask which project to search; do not silently choose a project.

1. Search for the user's question with:

   ```bash
   local-docs-rag search PROJECT --limit 8 "the user's question"
   ```

2. Search results are restricted to the specified project. Never combine or
   infer evidence across projects. Base the response only on the passages returned
   by the search. Cite each
   supporting passage using its source path and locator, formatted as
   `path#locator` (for example, `manual.pdf#page=4`).
3. If the search reports no matching documents, say that the local index did not
   return supporting evidence. Do not answer from general knowledge as though it
   came from the user's documents.
4. If the CLI is unavailable, tell the user to install the project and make its
   executable available in `PATH`. Do not switch to an MCP server or silently
   use another retrieval method.

Use `search`, not `ask`: Copilot should synthesize the retrieved passages and
citations. Do not run `index PROJECT` unless the user asks to refresh that
project's document index.
