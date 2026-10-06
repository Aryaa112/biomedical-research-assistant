from pathlib import Path

import pymupdf

from app.ingestion.models import PageRecord
from app.ingestion.quality import calculate_quality_score


def load_with_layout(pdf_path: Path) -> list[PageRecord]:
    """
    Layout-aware fallback.

    Extracts text blocks and orders them using their
    page coordinates.
    """

    pages: list[PageRecord] = []

    with pymupdf.open(pdf_path) as document:

        for page_index, page in enumerate(document):

            blocks = page.get_text("blocks")

            # Sort blocks top-to-bottom, then left-to-right.
            blocks = sorted(
                blocks,
                key=lambda block: (block[1], block[0]),
            )

            text_parts = []

            for block in blocks:
                block_text = block[4].strip()

                if block_text:
                    text_parts.append(block_text)

            text = "\n\n".join(text_parts)

            score = calculate_quality_score(text)

            pages.append(
                PageRecord(
                    page_number=page_index + 1,
                    text=text,
                    extraction_method="pymupdf_layout",
                    quality_score=score,
                )
            )

    return pages