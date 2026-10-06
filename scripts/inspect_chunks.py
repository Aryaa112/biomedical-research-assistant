import json
from pathlib import Path


CHUNK_FILE = Path("data/chunks/chunks.json")


def inspect_strategy(chunks, strategy, num_chunks=5):

    print()
    print("=" * 80)
    print(f"{strategy.upper()} CHUNKING")
    print("=" * 80)

    selected = [
        chunk
        for chunk in chunks
        if chunk["chunking_strategy"] == strategy
    ]

    print(f"Total chunks: {len(selected)}")

    for chunk in selected[:num_chunks]:

        print()
        print("-" * 80)
        print(f"Document: {chunk['filename']}")
        print(f"Page:     {chunk['page_number']}")
        print(f"Index:    {chunk['chunk_index']}")
        print(f"Length:   {len(chunk['source_text'])} characters")
        print("-" * 80)

        print(chunk["source_text"])


def main():

    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed = data["fixed"]
    recursive = data["recursive"]

    inspect_strategy(
        fixed,
        "fixed",
        num_chunks=5,
    )

    inspect_strategy(
        recursive,
        "recursive",
        num_chunks=5,
    )


if __name__ == "__main__":
    main()

    