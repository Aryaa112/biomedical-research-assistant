from app.retrieval.vector_store import (
    BiomedicalVectorStore,
)


def main():

    print("=" * 70)
    print("D4 VECTOR STORE BUILD")
    print("=" * 70)

    store = BiomedicalVectorStore()

    store.build()

    print()
    print("=" * 70)
    print("D4 VECTOR STORE BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()