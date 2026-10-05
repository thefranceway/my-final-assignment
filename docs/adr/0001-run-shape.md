# ADR 0001: Run shape

## Decision

Use a bounded loop for the research assistant.

The loop keeps the research task in one controlled execution path: retrieve
relevant documents, provide the retrieved context to the model, validate the
structured answer, and refuse when the required conditions are not met.

## Why

A graph would add additional model calls and transitions without evidence that
the final assignment needs multiple specialized roles. The loop is simpler to
trace, easier to bound, and consistent with the requirement for one model call
plus one corrective retry before refusal.

## Reversal measurement

Reconsider this decision if measured evaluation results show that a graph
materially improves critical-question performance enough to justify its
additional model calls and complexity.

## Reversal measurement

The loop decision should be reconsidered if a graph implementation improves
the critical-question pass rate by a meaningful measured margin while staying
within the assignment's model-call budget.
