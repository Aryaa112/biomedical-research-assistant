# app/guardrails/grounding_guardrail.py

"""
Grounding guardrail.

Checks whether the generated answer is reasonably grounded
in the text of the cited/retrieved chunks.

This is intentionally lightweight and deterministic.

A more advanced implementation can later use an LLM-based
verification pass.
"""

import re


# ----------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------

def _normalize_text(text: str) -> str:

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _extract_keywords(
    text: str,
) -> set[str]:

    normalized = _normalize_text(text)

    words = normalized.split()

    # Ignore very common words
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "was",
        "were",
        "are",
        "in",
        "on",
        "of",
        "to",
        "and",
        "or",
        "for",
        "with",
        "this",
        "that",
        "what",
        "how",
        "did",
        "used",
    }

    return {
        word
        for word in words
        if len(word) >= 4
        and word not in stopwords
    }


# ----------------------------------------------------------------------
# Grounding check
# ----------------------------------------------------------------------

def check_grounding(
    question: str,
    answer: str,
    cited_chunks: list[dict],
) -> tuple[str, str]:
    """
    Perform a lightweight grounding check.

    Returns:

        GROUNDED
        UNGROUNDED
        PARTIAL
    """

    if not answer or not answer.strip():

        return (
            "UNGROUNDED",
            "The generated answer is empty.",
        )

    if not cited_chunks:

        return (
            "UNGROUNDED",
            "No cited chunks were available.",
        )

    # --------------------------------------------------------------
    # Combine cited source text
    # --------------------------------------------------------------

    source_text = " ".join(
        chunk.get("source_text", "")
        for chunk in cited_chunks
    )

    source_keywords = _extract_keywords(
        source_text
    )

    answer_keywords = _extract_keywords(
        answer
    )

    if not answer_keywords:

        return (
            "UNGROUNDED",
            "No meaningful content was found in the answer.",
        )

    overlap = (
        answer_keywords & source_keywords
    )

    overlap_ratio = (
        len(overlap)
        / len(answer_keywords)
    )

    # --------------------------------------------------------------
    # Decision rule
    # --------------------------------------------------------------

    if overlap_ratio >= 0.50:

        return (
            "GROUNDED",
            "The answer has sufficient lexical overlap with the cited literature.",
        )

    if overlap_ratio >= 0.20:

        return (
            "PARTIAL",
            "Only part of the answer appears to be supported by the cited literature.",
        )

    return (
        "UNGROUNDED",
        "The answer has insufficient overlap with the cited literature.",
    )