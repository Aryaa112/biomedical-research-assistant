"""
Inspect processed biomedical documents.

Usage:
    python -m scripts.inspect_processed
"""

import json
from pathlib import Path


PROCESSED_DIR = Path("data/processed")


def main():
    files = sorted(PROCESSED_DIR.glob("*.json"))

    print(f"Processed documents: {len(files)}")
    print("=" * 80)

    for file_path in files:

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                document = json.load(f)

        except Exception as e:
            print(f"\nFAILED TO READ: {file_path.name}")
            print(f"Error: {e}")
            continue

        print()
        print(f"FILE: {document.get('filename', file_path.name)}")
        print(f"ID: {document.get('document_id', 'N/A')[:12]}...")

        print(f"TITLE: {document.get('title', 'N/A')}")
        print(f"AUTHORS: {document.get('authors', 'N/A')}")
        print(f"YEAR: {document.get('year', 'N/A')}")
        print(f"SOURCE URL: {document.get('source_url', 'N/A')}")

        # Support either metadata field name.
        method = document.get(
            "ingestion_method",
            document.get("method", "N/A")
        )

        print(f"METHOD: {method}")
        print(f"STATUS: {document.get('status', 'N/A')}")

        pages = document.get("pages", [])

        print(f"PAGES: {len(pages)}")

        for page in pages[:2]:

            page_number = page.get(
                "page_number",
                page.get("page", "N/A")
            )

            text = page.get("text", "")

            quality = page.get(
                "quality_score",
                page.get("quality", "N/A")
            )

            extraction_method = page.get(
                "extraction_method",
                page.get("method", method)
            )

            print(
                f"  Page {page_number} | "
                f"quality={quality} | "
                f"method={extraction_method}"
            )

            preview = text[:300].replace("\n", " ")

            print(f"    {preview}...")

        print("-" * 80)


if __name__ == "__main__":
    main()