from app.retrieval.retriever import BiomedicalRetriever


def main():

    print("=" * 80)
    print("D4.1 RETRIEVAL SMOKE TEST")
    print("=" * 80)

    query = (
        "What sample size was used in the "
        "Kisumu Combined Ring Study?"
    )

    print(f"\nQuery:\n{query}")

    retriever = BiomedicalRetriever(
        top_k=5
    )

    results = retriever.retrieve(
        query
    )

    print()
    print("-" * 80)
    print("TOP 5 RETRIEVED CHUNKS")
    print("-" * 80)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(
            f"#{rank}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Document: "
            f"{result['document_id']}"
        )

        print(
            f"Page: "
            f"{result['page_number']}"
        )

        print(
            f"Chunk index: "
            f"{result['chunk_index']}"
        )

        print(
            f"Distance: "
            f"{result['distance']:.4f}"
        )

        print(
            "Text:"
        )

        print(
            result["source_text"][:500]
            .replace("\n", " ")
        )

    print()
    print("=" * 80)
    print("SMOKE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()