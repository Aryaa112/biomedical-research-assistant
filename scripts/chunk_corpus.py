import json
from pathlib import Path

from app.chunking.chunker import chunk_pages


PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/chunks")


def load_document(path: Path) -> dict:

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    json_files = sorted(PROCESSED_DIR.glob("*.json"))

    print(f"Found {len(json_files)} processed documents")

    if not json_files:
        print("No processed JSON files found.")
        return

    fixed_chunks = []
    recursive_chunks = []

    for path in json_files:

        document = load_document(path)

        document_id = document["document_id"]
        filename = document["filename"]
        pages = document["pages"]

        print()
        print("-" * 70)
        print(f"Processing: {filename}")
        print(f"Pages: {len(pages)}")

        fixed = chunk_pages(
            pages=pages,
            document_id=document_id,
            filename=filename,
            strategy="fixed",
        )

        recursive = chunk_pages(
            pages=pages,
            document_id=document_id,
            filename=filename,
            strategy="recursive",
        )

        fixed_chunks.extend(fixed)
        recursive_chunks.extend(recursive)

        print(f"Fixed chunks: {len(fixed)}")
        print(f"Recursive chunks: {len(recursive)}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "fixed": fixed_chunks,
        "recursive": recursive_chunks,
    }

    output_file = OUTPUT_DIR / "chunks.json"

    with open(output_file, "w", encoding="utf-8") as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("D3 CHUNKING COMPLETE")
    print("=" * 70)

    print(f"Documents:        {len(json_files)}")
    print(f"Fixed chunks:     {len(fixed_chunks)}")
    print(f"Recursive chunks: {len(recursive_chunks)}")
    print(f"Output:           {output_file}")


if __name__ == "__main__":
    main()