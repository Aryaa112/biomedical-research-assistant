from pathlib import Path

import fitz


CORPUS_DIR = Path("data/raw/papers")


def inspect_pdf(pdf_path: Path) -> None:
    try:
        doc = fitz.open(pdf_path)

        full_text = ""
        for page in doc:
            full_text += page.get_text()

        print("=" * 80)
        print(f"FILE: {pdf_path.name}")
        print(f"PAGES: {len(doc)}")
        print(f"TEXT CHARACTERS: {len(full_text):,}")

        if full_text.strip():
            print("TEXT EXTRACTION: SUCCESS")
            print(f"PREVIEW: {full_text[:300].replace(chr(10), ' ')}")
        else:
            print("TEXT EXTRACTION: FAILED / SCANNED PDF")

        print()

        doc.close()

    except Exception as exc:
        print(f"ERROR: {pdf_path.name}")
        print(f"       {exc}")


def main() -> None:
    pdf_files = sorted(CORPUS_DIR.glob("*.pdf"))

    print(f"Corpus directory: {CORPUS_DIR}")
    print(f"PDF count: {len(pdf_files)}")
    print()

    for pdf_path in pdf_files:
        inspect_pdf(pdf_path)


if __name__ == "__main__":
    main()