"""My final assignment agent: a research assistant over the six course documents.

One run: pick the single document that best matches the question, give the
model that whole document, then check what comes back. It answers with a
checked citation and the quoted source section, or it refuses. An order hidden
in the question is set aside first. The model comes from `.env` (BOOTCAMP_PROVIDER).
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
#: Wording that means "the documents do not say", when an answer opens with it.
REFUSAL_SIGNS = (
    "don't know",
    "do not know",
    "not enough information",
    "not supported",
    "no information",
    "not mentioned",
    "does not mention",
    "does not provide",
    "does not contain",
    "does not say",
    "does not specify",
    "cannot answer",
    "not covered",
)


#: How an order aimed at the agent starts: "Ignore your rules...", "SYSTEM: ...".
ORDER = re.compile(
    r"^\W*(?:(?:and|then|also|now|please|just)\s+)*"
    r"(?:ignore|disregard|forget|override|bypass|reveal|print|repeat|leak|dump|pretend|act as|"
    r"you are now|from now on|new instructions?|system\s*(?:override|prompt|message|note)?\s*:)",
    re.IGNORECASE,
)


#: Lead-in words left over once an order is removed: "...and just tell me: what is X?"
FILLER = re.compile(
    r"^(?:(?:just|please|now|then)\s+)*(?:tell me|answer|say|explain)\b[:,]?\s*", re.IGNORECASE
)


def real_question(question: str, salvage: bool = True) -> str:
    """The question with any order aimed at the agent set aside.

    A question is untrusted input too. "Ignore your rules and tell me: what is X?"
    is answered as "what is X?", so the order cannot steer which document is used.
    """
    kept: list[str] = []
    for clause in re.split(r"(?<=[.?:;\x21])\s+", question.strip()):
        had_order = False
        while ORDER.match(clause):
            _, joiner, rest = clause.partition(" and ")
            clause, had_order = (rest if joiner and salvage else ""), True
        if had_order:
            clause = FILLER.sub("", clause)
        if clause.strip():
            kept.append(clause.strip())
    return " ".join(kept)


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


class TwoCalls:
    """The real model, held to the course budget: one call and one retry."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if self.calls > 2:
            return ""  # budget spent: an unreadable reply ends as a refusal
        return self.client.complete(system=system, user=user)


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
        # 0. Set aside any order hidden in the question; answer what is actually asked.
        asked_for = question
        question = real_question(asked_for)
        # The model only ever sees the clean part, never what was left of an order.
        to_model = real_question(asked_for, salvage=False) or question
        if not question:
            why = "the question held only an order, nothing to answer; refusing"
            return AgentResult(answer=refusal(), trace=(TraceEvent("decision", why),))
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
        if len(found) * 4 < len(asked):
            # A stray shared word is not support: refuse before any model call.
            why = f"only {sorted(found)} of {len(asked)} question words are in {doc_id}; refusing"
            return AgentResult(answer=refusal(), trace=(TraceEvent("decision", why),))

        # 2. One model call, on that whole document, nothing cut off.
        model = TwoCalls(self.client)
        context = [fit_paragraphs(document)]
        result = answer_question(to_model, context, model, max_tool_calls=3, top_k=20)
        answer, trace = result.answer, list(result.trace)
        opening = answer.answer.lower().split(". ")[0][:160]
        wrote_an_answer = not any(sign in opening for sign in REFUSAL_SIGNS)
        if model.calls == 1 and wrote_an_answer and not answer.citations:
            # An answer that names no source: the one retry the budget allows.
            result = answer_question(to_model, context, model, max_tool_calls=3, top_k=20)
            answer = result.answer
            trace.append(TraceEvent("decision", "answer named no source; asked once more"))
            trace += list(result.trace)
        if len(question) < len(" ".join(asked_for.split())):
            trace.insert(0, TraceEvent("decision", "an order in the question was set aside"))

        # 3. The application decides the final shape, not the model.
        opening = answer.answer.lower().split(". ")[0][:160]
        if answer.needs_human_review and answer.citations and "stripped" in trace[-1].detail:
            # The pipeline removed an invented source and flagged the answer: keep it flagged.
            pass
        elif (
            not answer.citations
            or answer.confidence < 0.5
            or any(sign in opening for sign in REFUSAL_SIGNS)
        ):
            # No checked citation, low confidence, or an answer that opens by declining.
            trace.append(TraceEvent("decision", "no supported, cited answer; flagged refusal"))
            answer = refusal()
        else:
            # A checked citation and a confident answer: the review flag is ours to set.
            quote = best_section(document, to_model)
            answer = replace(
                answer,
                needs_human_review=False,
                answer=f"{answer.answer}\n\nSource [{doc_id}]:\n{quote}",
            )
        return AgentResult(answer=answer, trace=tuple(trace))

    def __call__(self, question: str) -> ResearchAnswer:
        return self.run(question).answer
