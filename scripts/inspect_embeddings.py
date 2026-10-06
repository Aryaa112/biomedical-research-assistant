import json
from pathlib import Path

import numpy as np


EMBEDDINGS_FILE = Path("data/embeddings/embeddings.json")
TOP_K = 5


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def main():

    print("=" * 80)
    print("D3 EMBEDDING SANITY CHECK")
    print("=" * 80)

    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["embeddings"]

    print(f"Total embeddings: {len(chunks)}")

    # Pick 3 representative chunks
    selected_indices = [0, len(chunks) // 2, len(chunks) - 1]

    for query_index in selected_indices:

        query_chunk = chunks[query_index]
        query_embedding = query_chunk["embedding"]

        similarities = []

        for i, candidate in enumerate(chunks):

            if i == query_index:
                continue

            score = cosine_similarity(
                query_embedding,
                candidate["embedding"]
            )

            similarities.append((score, i, candidate))

        similarities.sort(
            key=lambda x: x[0],
            reverse=True
        )

        print()
        print("=" * 80)
        print(f"QUERY CHUNK #{query_index}")
        print("=" * 80)

        print(
            f"Document: {query_chunk['document_id']}"
        )
        print(
            f"Page: {query_chunk['page_number']}"
        )

        print("\nQuery text:")
        print(query_chunk["source_text"][:500])

        print()
        print("-" * 80)
        print("TOP 5 NEAREST NEIGHBOURS")
        print("-" * 80)

        for rank, (score, index, candidate) in enumerate(
            similarities[:TOP_K],
            start=1
        ):

            print()
            print(f"#{rank} | similarity={score:.4f}")
            print(
                f"Document: {candidate['document_id']}"
            )
            print(
                f"Page: {candidate['page_number']}"
            )
            print(
                f"Chunk index: {candidate['chunk_index']}"
            )

            print(
                candidate["source_text"][:400]
                .replace("\n", " ")
            )

    print()
    print("=" * 80)
    print("SANITY CHECK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()