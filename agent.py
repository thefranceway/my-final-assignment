"""My final assignment agent: a research assistant over the six course documents.

One run: pick the single document that best matches the question, give the
model that whole document, then check what comes back. It answers with a
citation and the quoted source section, or it refuses. The model comes from
`.env` (BOOTCAMP_PROVIDER).
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from bootcamp_agent.agent import AgentResult, TraceEvent, answer_question
from bootcamp_agent.config import load_settings
from bootcamp_agent.documents import Document, load_corpus
from bootcamp_agent.llm import LLMClient, get_client
from bootcamp_agent.retrieval import _tokens, retrieve
from bootcamp_agent.schema import ResearchAnswer
from bootcamp_agent.tools import Tool, build_tools

#: The six course documents. Versioned input: nothing here writes to it.
CORPUS_DIR = Path(__file__).resolve().parent / "data" / "corpus"

REFUSAL_TEXT = "I don't know based on the provided corpus."
REFUSAL_SIGNS = ("don't know", "do not know", "not enough information", "not supported")

#: Added to the question the model sees. It never changes which document is used.
ANSWER_STYLE = (
    "Answer completely: when the context lists several items, include every one. "
    "Keep the context's own wording for key terms. Cite the doc-id in square brackets."
)


def fit_paragraphs(doc: Document, limit: int = 700) -> Document:
    """The same document, with long paragraphs split at sentence ends.

    The course chunker cuts a paragraph at 800 characters and drops the rest,
    so the end of a long paragraph never reached the model.
    """
    paragraphs: list[str] = []
    for paragraph in doc.text.split("\n\n"):
        piece = ""
        for sentence in re.split(r"(?<=[.?:])\s+", paragraph.strip()):
            if piece and len(piece) + len(sentence) + 1 > limit:
                paragraphs.append(piece)
                piece = sentence
            else:
                piece = f"{piece} {sentence}".strip()
        if piece:
            paragraphs.append(piece)
    return replace(doc, text="\n\n".join(paragraphs))


def best_section(doc: Document, question: str) -> str:
    """The section of the document that shares the most words with the question."""
    asked = set(_tokens(question))
    sections = re.split(r"\n(?=#+ )", doc.text)
    words = [set(_tokens(part)) for part in sections]

    def score(index: int) -> float:
        # A word found in few sections says more than a word found in all of them.
        return sum(1 / sum(word in other for other in words) for word in asked & words[index])

    return sections[max(range(len(sections)), key=score)].strip()


def refusal() -> ResearchAnswer:
    return ResearchAnswer(
        answer=REFUSAL_TEXT, citations=(), confidence=0.0, needs_human_review=True
    )


class YourAgent:
    """The agent the tests and the grader run."""

    #: Not enforced yet (the `timeout` contract test is still xfail).
    timeout_s: float = 30.0

    def __init__(self, client: LLMClient | None = None) -> None:
        self.documents: list[Document] = load_corpus(CORPUS_DIR)
        self.client: LLMClient = client if client is not None else get_client(load_settings())
        self.tools: dict[str, Tool] = build_tools(self.documents, self.client)

    def run(self, question: str) -> AgentResult:
        """One question, answered or refused, with the trace of how."""
        # 1. Rank whole documents, not chunks, and keep the best one.
        ranked = retrieve(question, self.documents, top_k=6, max_chars=1_000_000)
        if not ranked:
            # Nothing matches: the pipeline refuses without a model call.
            return answer_question(question, self.documents, self.client, max_tool_calls=3, top_k=3)
        doc_id = ranked[0].chunk.doc_id
        close = {hit.chunk.doc_id for hit in ranked if hit.score >= 0.9 * ranked[0].score}
        if len(close) > 1:
            # A near tie between documents: the best single passage decides.
            for hit in retrieve(question, self.documents, top_k=50):
                if hit.chunk.doc_id in close:
                    doc_id = hit.chunk.doc_id
                    break
        document = next(doc for doc in self.documents if doc.doc_id == doc_id)
        asked = set(_tokens(question))
        found = asked & set(_tokens(document.text))
        if len(found) * 3 < len(asked):
            # A stray shared word is not support: refuse before any model call.
            why = f"only {sorted(found)} of {len(asked)} question words are in {doc_id}; refusing"
            return AgentResult(answer=refusal(), trace=(TraceEvent("decision", why),))

        # 2. One model call, on that whole document, nothing cut off.
        result = answer_question(
            f"{question}\n\n{ANSWER_STYLE}",
            [fit_paragraphs(document)],
            self.client,
            max_tool_calls=3,
            top_k=20,
        )
        answer, trace = result.answer, list(result.trace)

        # 3. The application decides the final shape, not the model.
        text = answer.answer.lower()
        if any(sign in text for sign in REFUSAL_SIGNS):
            trace.append(TraceEvent("decision", "the model did not know; flagged refusal"))
            answer = refusal()
        elif not answer.needs_human_review:
            if not answer.citations:
                trace.append(TraceEvent("decision", f"answer came from {doc_id} only; cited it"))
                answer = replace(answer, citations=(doc_id,))
            # Show the evidence: quote the section the answer rests on, word for word.
            quote = best_section(document, question)
            answer = replace(answer, answer=f"{answer.answer}\n\nSource [{doc_id}]:\n{quote}")
        return AgentResult(answer=answer, trace=tuple(trace))

    def __call__(self, question: str) -> ResearchAnswer:
        return self.run(question).answer
