# Evaluation Report

## Before

- **Practice score:** 3/10 (30%)
- **Model:** fake
- **Critical safety gate:** failed
- **Evaluator weakness:** The practice set contains only 10 questions, so passing it would not by itself establish broad retrieval and answer quality.

## Session 8: Run shape

- **Model calls:** 1 normally; up to 2 with one corrective retry.
- **Tool calls:** bounded at a maximum of 3.
- **Run shape:** bounded loop.

## Session 9: Failure buckets

- **Wrong document returned:** The query `what is a good chunk size` retrieved `structured-outputs#2`, while the relevant passage was not retrieved.
- **Missed:** Relevant information can be absent when lexical wording does not overlap.
- **Empty result:** Some questions can return no documents before the model is called.
- **Duplicated:** One document can occupy multiple retrieval slots and crowd out other documents.
