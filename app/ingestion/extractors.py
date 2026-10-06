from pathlib import Path

import pymupdf
import pdfplumber

from .quality import calculate_text_quality, is_garbled

import pytesseract
from PIL import Image


def extract_with_pymupdf(pdf_path: Path):
    """
    Primary extractor.

    Preserves page boundaries and is fast.
    """

    pages = []

    with pymupdf.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text")

            quality = calculate_text_quality(text)

            pages.append({
                "page_number": page_number,
                "text": text,
                "method": "pymupdf",
                "quality": quality,
            })

    return pages


def extract_with_pdfplumber(pdf_path: Path):
    """
    Secondary extractor.

    Useful when PyMuPDF extraction is poor and
    can handle some PDF layouts/tables differently.
    """

    pages = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf.pages, start=1):

            text = page.extract_text() or ""

            quality = calculate_text_quality(text)

            pages.append({
                "page_number": page_number,
                "text": text,
                "method": "pdfplumber",
                "quality": quality,
            })

    return pages


def extract_tables_with_pdfplumber(pdf_path: Path):
    """
    Extract tables separately.

    Tables are stored as structured rows rather than
    being blindly mixed into normal text.
    """

    tables = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf.pages, start=1):

            page_tables = page.extract_tables()

            for table_index, table in enumerate(page_tables):

                tables.append({
                    "page_number": page_number,
                    "table_index": table_index,
                    "rows": table,
                })

    return tables




def extract_with_ocr(pdf_path: Path):
    """
    OCR fallback for scanned/image PDFs.
    """

    pages = []

    with pymupdf.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            # Render PDF page as image.
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))

            image = Image.frombytes(
                "RGB",
                [pixmap.width, pixmap.height],
                pixmap.samples,
            )

            text = pytesseract.image_to_string(image)

            quality = calculate_text_quality(text)

            pages.append({
                "page_number": page_number,
                "text": text,
                "method": "ocr",
                "quality": quality,
            })

    return pages