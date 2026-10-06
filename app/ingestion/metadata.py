import re
from pathlib import Path

import pymupdf


def extract_metadata(pdf_path: Path) -> dict:
    """Extract basic metadata embedded in the PDF."""

    with pymupdf.open(pdf_path) as document:
        metadata = document.metadata

    title = metadata.get("title") or None
    author_string = metadata.get("author") or ""

    authors = [
        author.strip()
        for author in re.split(r"[;,]", author_string)
        if author.strip()
    ]

    year = None

    if metadata.get("creationDate"):
        match = re.search(
            r"(19|20)\d{2}",
            metadata["creationDate"],
        )

        if match:
            year = int(match.group())

    return {
        "title": title,
        "authors": authors,
        "year": year,
    }