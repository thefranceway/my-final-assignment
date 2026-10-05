# Research Assistant Skill

## Before

The research assistant retrieves documents using lexical word overlap and passes
the retrieved context to the model. A trace for:

`what is a good chunk size`

returned `structured-outputs#2` and the agent refused with no citation.

## After

The research assistant should use the retrieved document context to produce a
grounded `ResearchAnswer` with a citation when the corpus supports the answer,
and refuse when the retrieved evidence does not support an answer.

### Verification

Run:

`uv run bootcamp final trace "what is a good chunk size"`

The after run should return a grounded answer with a citation to the document
that actually supports the answer.

## Safety boundary

| Tool | Access |
|---|---|
| `search_documents` | READ |
| `get_document_metadata` | READ |
| `summarize_document` | READ |

Only read-only tools are wired to the final assignment. No writing tool is wired.
