from pathlib import Path

from app.ingestion.pipeline import (
    ingest_document,
    save_document,
)


RAW_DIR = Path("data/raw/papers")
PROCESSED_DIR = Path("data/processed")


def main():

    pdf_files = sorted(
        RAW_DIR.glob("*.pdf")
    )

    print(f"Found {len(pdf_files)} PDFs")

    for pdf_path in pdf_files:

        print("\n" + "-" * 70)
        print(f"Processing: {pdf_path.name}")

        try:

            document = ingest_document(pdf_path)

            save_document(
                document,
                PROCESSED_DIR,
            )

            print(
                f"ID: {document.document_id[:12]}..."
            )

            print(
                f"Method: {document.extraction_method}"
            )

            print(
                f"Pages: {len(document.pages)}"
            )

            print("Status: SUCCESS")

            # Show quality for each page.
            for page in document.pages:

                print(
                    f"  Page {page.page_number}: "
                    f"quality={page.quality_score:.3f} "
                    f"method={page.extraction_method}"
                )

        except Exception as exc:

            print(
                f"FAILED: {exc}"
            )


if __name__ == "__main__":
    main()