from pathlib import Path

import pymupdf

from app.ingestion.models import PageRecord
from app.ingestion.quality import calculate_quality_score


def load_with_pymupdf(pdf_path: Path) -> list[PageRecord]:
    """
    Extract a PDF page-by-page using PyMuPDF.
    """

    pages: list[PageRecord] = []

    with pymupdf.open(pdf_path) as document:

        for page_index, page in enumerate(document):
            text = page.get_text("text").strip()

            score = calculate_quality_score(text)

            pages.append(
                PageRecord(
                    page_number=page_index + 1,
                    text=text,
                    extraction_method="pymupdf",
                    quality_score=score,
                )
            )

    return pages