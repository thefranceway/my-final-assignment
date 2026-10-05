# Ranked issues

## Session 9

| rank | issue | impact |
|---:|---|---|
| 1 | Retrieval returned the wrong document for a question about chunk size, so the relevant passage did not reach the model. | The agent cannot reliably answer grounded questions when lexical retrieval selects an irrelevant source. |
| 2 | The current retrieval method relies on shared words and can miss relevant passages when wording differs. | Paraphrased questions can fail even when the corpus contains the answer. |
| 3 | The agent refuses after receiving insufficient context instead of recovering the relevant source. | Users receive unnecessary refusals for questions that are answerable from the corpus. |

## Rank 1, in progress

- **Trace:** `[retrieve] top_k=3 -> [('structured-outputs', 2)]`
- **Failure bucket:** wrong document returned
