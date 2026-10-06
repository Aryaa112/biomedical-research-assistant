# app/guardrails/retrieval_guardrail.py

"""
Retrieval guardrail.

Responsibilities:

1. Reject empty retrieval.
2. Reject weak/out-of-corpus retrieval.
3. Rank chunks by similarity.
4. Keep only strongest chunks.
5. Preserve citation metadata.
6. Limit source text size.
"""

from typing import Any


# ======================================================================
# CONFIGURATION
# ======================================================================

DEFAULT_DISTANCE_THRESHOLD = 0.70

DEFAULT_MAX_CHUNKS = 2

DEFAULT_MAX_CHARS_PER_CHUNK = 1200


# ======================================================================
# RESULT
# ======================================================================

class RetrievalGuardrailResult:

    def __init__(
        self,
        allowed: bool,
        reason: str,
        relevant_chunks: list[dict[str, Any]],
    ):

        self.allowed = allowed
        self.reason = reason
        self.relevant_chunks = relevant_chunks

    def __repr__(self) -> str:

        return (
            "RetrievalGuardrailResult("
            f"allowed={self.allowed}, "
            f"reason={self.reason!r}, "
            f"chunks={len(self.relevant_chunks)})"
        )


# ======================================================================
# COMPACT CHUNK
# ======================================================================

def _compact_chunk(
    chunk: dict[str, Any],
    max_chars: int,
) -> dict[str, Any]:

    source_text = str(
        chunk.get(
            "source_text",
            "",
        )
    )

    if len(source_text) > max_chars:

        source_text = (
            source_text[:max_chars]
            + "\n[Source text truncated.]"
        )

    return {
        "chunk_id": chunk.get("chunk_id"),
        "document_id": chunk.get("document_id"),
        "page_number": chunk.get("page_number"),
        "source_text": source_text,
        "distance": chunk.get("distance"),
    }


# ======================================================================
# MAIN GUARDRAIL
# ======================================================================

def check_retrieval(
    chunks: list[dict[str, Any]],
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
) -> RetrievalGuardrailResult:

    # --------------------------------------------------------------
    # Configuration validation
    # --------------------------------------------------------------

    if max_chunks <= 0:

        raise ValueError(
            "max_chunks must be greater than zero."
        )

    if max_chars_per_chunk <= 0:

        raise ValueError(
            "max_chars_per_chunk must be greater than zero."
        )

    # --------------------------------------------------------------
    # Empty retrieval
    # --------------------------------------------------------------

    if not chunks:

        return RetrievalGuardrailResult(
            allowed=False,
            reason=(
                "No literature chunks were retrieved. "
                "The question may be outside the available corpus."
            ),
            relevant_chunks=[],
        )

    # --------------------------------------------------------------
    # Filter invalid chunks
    # --------------------------------------------------------------

    valid_chunks = []

    for chunk in chunks:

        if not isinstance(chunk, dict):
            continue

        distance = chunk.get("distance")

        if distance is None:
            continue

        try:

            distance = float(distance)

        except (
            TypeError,
            ValueError,
        ):

            continue

        if distance <= distance_threshold:

            valid_chunks.append(
                chunk
            )

    # --------------------------------------------------------------
    # No relevant chunks
    # --------------------------------------------------------------

    if not valid_chunks:

        return RetrievalGuardrailResult(
            allowed=False,
            reason=(
                "Retrieved chunks did not meet the relevance "
                "threshold. The information was not found "
                "in the available literature."
            ),
            relevant_chunks=[],
        )

    # --------------------------------------------------------------
    # Sort by distance
    # Lower distance = better
    # --------------------------------------------------------------

    valid_chunks.sort(
        key=lambda chunk: float(
            chunk.get(
                "distance",
                999999,
            )
        )
    )

    # --------------------------------------------------------------
    # Select strongest chunks
    # --------------------------------------------------------------

    selected_chunks = valid_chunks[
        :max_chunks
    ]

    # --------------------------------------------------------------
    # Compact
    # --------------------------------------------------------------

    compact_chunks = [

        _compact_chunk(
            chunk=chunk,
            max_chars=max_chars_per_chunk,
        )

        for chunk in selected_chunks
    ]

    return RetrievalGuardrailResult(
        allowed=True,
        reason=(
            "Sufficiently relevant literature was retrieved. "
            f"Using {len(compact_chunks)} strongest chunks."
        ),
        relevant_chunks=compact_chunks,
    )