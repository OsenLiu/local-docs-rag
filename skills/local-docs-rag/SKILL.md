---
name: local-docs-rag
description: 'Search and answer questions from the local PDF and Markdown document index. Use when a user asks about their local documents or requests evidence from indexed files.'
---

# Local document search

Use the `local-docs-rag` CLI to retrieve evidence, then answer using Copilot.

1. Search for the user's question with:

   ```bash
   local-docs-rag search --limit 8 "the user's question"
   ```

2. Base the response only on the passages returned by the search. Cite each
   supporting passage using its source path and locator, formatted as
   `path#locator` (for example, `manual.pdf#page=4`).
3. If the search reports no matching documents, say that the local index did not
   return supporting evidence. Do not answer from general knowledge as though it
   came from the user's documents.
4. If the CLI is unavailable, tell the user to install the project and make its
   executable available in `PATH`. Do not switch to an MCP server or silently
   use another retrieval method.

Use `search`, not `ask`: Copilot should synthesize the retrieved passages and
citations. Do not run `index` unless the user asks to refresh the document index.
