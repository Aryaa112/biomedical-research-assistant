#app\chunking\chunker.py


import hashlib
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# Configuration
# ============================================================

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


# ============================================================
# Utility functions
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean PDF-extracted text while preserving paragraph structure.
    """

    if not text:
        return ""

    # Normalize non-breaking spaces
    text = text.replace("\xa0", " ")

    # Normalize excessive spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Preserve paragraphs but remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def get_content_hash(text: str) -> str:
    """
    Stable hash used later for embedding caching.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# FIXED-SIZE CHUNKING
# ============================================================

def fixed_chunk_page(
    text: str,
    document_id: str,
    page_number: int,
    filename: str,
):
    """
    Fixed-size character chunking.

    Baseline strategy:
        chunk_size = 2000 characters
        overlap = 200 characters
    """

    text = clean_text(text)

    results = []

    if not text:
        return results

    start = 0
    chunk_index = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:

            results.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "source_text": chunk,
                    "content_hash": get_content_hash(chunk),
                    "chunking_strategy": "fixed",
                }
            )

            chunk_index += 1

        # Move forward while retaining overlap
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return results


# ============================================================
# RECURSIVE CHUNKING
# ============================================================

def recursive_chunk_page(
    text: str,
    document_id: str,
    page_number: int,
    filename: str,
):
    """
    Recursive / structure-aware chunking.

    Splitting priority:

        paragraph
             ↓
        line
             ↓
        sentence
             ↓
        smaller punctuation
             ↓
        word
             ↓
        character

    This allows the splitter to preserve meaningful
    boundaries whenever possible.
    """

    text = clean_text(text)

    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",   # paragraph
            "\n",     # line
            ". ",     # sentence
            "? ",
            "! ",
            "; ",
            ", ",
            " ",      # word
            "",       # character fallback
        ],

        length_function=len,

        is_separator_regex=False,
    )

    chunks = splitter.split_text(text)

    results = []

    for index, chunk in enumerate(chunks):

        chunk = chunk.strip()

        if not chunk:
            continue

        results.append(
            {
                "document_id": document_id,
                "filename": filename,
                "page_number": page_number,
                "chunk_index": index,
                "source_text": chunk,
                "content_hash": get_content_hash(chunk),
                "chunking_strategy": "recursive",
            }
        )

    return results


# ============================================================
# MAIN CHUNKING FUNCTION
# ============================================================

def chunk_pages(
    pages,
    document_id: str,
    filename: str,
    strategy: str,
):
    """
    Chunk every page of a document.

    Parameters
    ----------
    pages:
        List of page objects from processed JSON.

    document_id:
        Stable document identifier.

    filename:
        Original PDF filename.

    strategy:
        Either:
            "fixed"
            "recursive"

    Returns
    -------
    list:
        List of chunk dictionaries.
    """

    if strategy not in {"fixed", "recursive"}:

        raise ValueError(
            f"Unknown chunking strategy: {strategy}. "
            "Use 'fixed' or 'recursive'."
        )

    all_chunks = []

    for page in pages:

        page_number = page.get("page_number")

        # Your ingestion JSON should normally use "text".
        # source_text is included as a fallback.
        text = page.get("text", "")

        if not text:
            text = page.get("source_text", "")

        if not text:
            continue

        # ----------------------------------------------------
        # Fixed
        # ----------------------------------------------------

        if strategy == "fixed":

            chunks = fixed_chunk_page(
                text=text,
                document_id=document_id,
                page_number=page_number,
                filename=filename,
            )

        # ----------------------------------------------------
        # Recursive
        # ----------------------------------------------------

        else:

            chunks = recursive_chunk_page(
                text=text,
                document_id=document_id,
                page_number=page_number,
                filename=filename,
            )

        all_chunks.extend(chunks)

    return all_chunks