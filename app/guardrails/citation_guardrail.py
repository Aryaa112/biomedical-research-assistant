# app/guardrails/citation_guardrail.py

"""
Citation enforcement guardrail.

Ensures that:
    1. A non-empty answer has citations.
    2. Citations refer to actually retrieved chunks.
    3. The cited document/page/chunk combination is valid.
"""


def validate_citations(
    response,
    retrieved_chunks: list[dict],
) -> tuple[bool, str]:
    """
    Validate citations returned by the RAG response.

    Args:
        response:
            Structured RAGResponse.

        retrieved_chunks:
            Chunks actually returned by retrieval.

    Returns:
        (valid, reason)
    """

    answer = getattr(
        response,
        "answer",
        "",
    )

    citations = getattr(
        response,
        "citations",
        [],
    )

    # --------------------------------------------------------------
    # Empty answer
    # --------------------------------------------------------------

    if not answer or not answer.strip():

        return (
            False,
            "The model returned an empty answer.",
        )

    # --------------------------------------------------------------
    # Answer without citations
    # --------------------------------------------------------------

    if not citations:

        return (
            False,
            "Answer contains factual content but no citations.",
        )

    # --------------------------------------------------------------
    # Build valid citation set
    # --------------------------------------------------------------

    valid_citations = set()

    for chunk in retrieved_chunks:

        valid_citations.add(
            (
                chunk.get("document_id"),
                chunk.get("page_number"),
                chunk.get("chunk_id"),
            )
        )

    # --------------------------------------------------------------
    # Validate every citation
    # --------------------------------------------------------------

    for citation in citations:

        citation_tuple = (
            getattr(
                citation,
                "document_id",
                None,
            ),
            getattr(
                citation,
                "page_number",
                None,
            ),
            getattr(
                citation,
                "chunk_id",
                None,
            ),
        )

        if citation_tuple not in valid_citations:

            return (
                False,
                (
                    "The model returned a citation that does "
                    "not correspond to a retrieved chunk."
                ),
            )

    # --------------------------------------------------------------
    # Everything valid
    # --------------------------------------------------------------

    return (
        True,
        "All citations are valid and correspond to retrieved chunks.",
    )