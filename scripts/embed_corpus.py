import json
from pathlib import Path

from app.embeddings.embedder import EmbeddingStore


CHUNKS_FILE = Path("data/chunks/chunks.json")
OUTPUT_FILE = Path("data/embeddings/embeddings.json")


def load_recursive_chunks(path: Path) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    return data["recursive"]


def main():

    print("=" * 70)
    print("D3 EMBEDDING PIPELINE")
    print("=" * 70)

    chunks = load_recursive_chunks(CHUNKS_FILE)

    print(f"Found {len(chunks)} recursive chunks")

    if not chunks:
        print("No chunks found.")
        return

    store = EmbeddingStore()

    store.embed_chunks(
        chunks=chunks,
        output_path=OUTPUT_FILE,
    )

    print()
    print("=" * 70)
    print("D3 EMBEDDING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()