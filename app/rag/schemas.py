# app/rag/schemas.py

from typing import Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """
    Identifies the exact literature chunk used to support an answer.
    """

    document_id: str = Field(
        description="Unique identifier of the source document."
    )

    page_number: int = Field(
        description="Page number where the supporting information appears."
    )

    chunk_id: str = Field(
        description="Unique identifier of the retrieved chunk."
    )


class RAGResponse(BaseModel):
    """
    Structured output produced by the generation layer.
    """

    answer: str = Field(
        description=(
            "Answer to the user's question based only on the "
            "retrieved biomedical literature."
        )
    )

    citations: list[Citation] = Field(
        description=(
            "Literature chunks that directly support the answer. "
            "Every factual answer must have at least one citation."
        )
    )

    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence that the retrieved literature sufficiently "
            "supports the answer."
        )
    )