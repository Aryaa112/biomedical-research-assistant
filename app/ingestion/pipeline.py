import hashlib
import json
from pathlib import Path

from .models import Document, PageContent
from .extractors import (
    extract_with_pymupdf,
    extract_with_pdfplumber,
    extract_with_ocr,
)


QUALITY_THRESHOLD = 0.70


def generate_document_id(pdf_path: Path) -> str:
    """
    Stable document ID based on file contents.
    """

    sha256 = hashlib.sha256()

    with open(pdf_path, "rb") as file:

        while chunk := file.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()


def choose_best_extraction(pdf_path: Path):

    # --------------------------------------------------
    # 1. PRIMARY: PyMuPDF
    # --------------------------------------------------

    pymupdf_pages = extract_with_pymupdf(pdf_path)

    good_pages = [
        page
        for page in pymupdf_pages
        if page["quality"] >= QUALITY_THRESHOLD
    ]

    if len(good_pages) == len(pymupdf_pages):

        return pymupdf_pages, "pymupdf"

    # --------------------------------------------------
    # 2. FALLBACK: pdfplumber
    # --------------------------------------------------

    pdfplumber_pages = extract_with_pdfplumber(pdf_path)

    good_pages = [
        page
        for page in pdfplumber_pages
        if page["quality"] >= QUALITY_THRESHOLD
    ]

    if len(good_pages) >= len(pymupdf_pages) * 0.8:

        return pdfplumber_pages, "pdfplumber"

    # --------------------------------------------------
    # 3. LAST RESORT: OCR
    # --------------------------------------------------

    ocr_pages = extract_with_ocr(pdf_path)

    return ocr_pages, "ocr"


def extract_title(text: str, filename: str) -> str:

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if lines:
        return lines[0][:500]

    return filename


def ingest_document(pdf_path: Path) -> Document:

    document_id = generate_document_id(pdf_path)

    pages, method = choose_best_extraction(pdf_path)

    page_objects = []

    for page in pages:

        page_objects.append(
            PageContent(
                page_number=page["page_number"],
                text=page["text"],
                extraction_method=page["method"],
                quality_score=page["quality"],
                needs_ocr=page["quality"] < QUALITY_THRESHOLD,
            )
        )

    combined_text = "\n".join(
        page.text
        for page in page_objects
    )

    title = extract_title(
        combined_text,
        pdf_path.name,
    )

    return Document(
        document_id=document_id,
        filename=pdf_path.name,
        title=title,
        authors=[],
        year=None,
        source_url=None,
        pages=page_objects,
        extraction_method=method,
        status="success",
    )


def save_document(document: Document, output_dir: Path):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / f"{document.document_id}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            document.to_dict(),
            file,
            indent=2,
            ensure_ascii=False,
        )