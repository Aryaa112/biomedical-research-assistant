import sys
import hashlib
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"


def get_content_hash(text: str) -> str:
    """
    Generate a stable hash for a chunk's text.

    If the text has not changed, its hash remains the same.
    This allows us to reuse an existing embedding.
    """
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


class EmbeddingStore:

    def __init__(self, model_name: str = MODEL_NAME):

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print("Embedding model loaded.")

    def load_existing(self, path: Path) -> dict:
        """
        Load previously persisted embeddings.

        Returns:
            Dictionary:
                content_hash -> embedding record
        """

        if not path.exists():
            return {}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        existing = {}

        for item in data.get("embeddings", []):
            existing[item["content_hash"]] = item

        return existing

    def embed_chunks(
        self,
        chunks: list[dict],
        output_path: Path,
    ) -> dict:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        existing = self.load_existing(output_path)

        results = []

        embedded_count = 0
        cached_count = 0

        print()
        print("=" * 70)
        print("EMBEDDING CORPUS")
        print("=" * 70)

        for i, chunk in enumerate(chunks):

            source_text = chunk["source_text"]

            content_hash = get_content_hash(source_text)

            # --------------------------------------------------
            # CACHE CHECK
            # --------------------------------------------------

            if content_hash in existing:

                record = existing[content_hash]

                results.append(record)

                cached_count += 1

                continue

            # --------------------------------------------------
            # CREATE NEW EMBEDDING
            # --------------------------------------------------

            embedding = self.model.encode(
                source_text,
                normalize_embeddings=True,
            )

            record = {
                "document_id": chunk["document_id"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "source_text": source_text,
                "content_hash": content_hash,
                "chunking_strategy": chunk.get(
                    "chunking_strategy",
                    "recursive",
                ),
                "embedding": embedding.tolist(),
            }

            results.append(record)

            embedded_count += 1

            if embedded_count % 10 == 0:
                print(
                    f"Embedded: {embedded_count}"
                )

        # ------------------------------------------------------
        # SAVE
        # ------------------------------------------------------

        output = {
            "model": MODEL_NAME,
            "embedding_dimension": len(
                results[0]["embedding"]
            ) if results else 0,
            "embeddings": results,
        }

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                output,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print("-" * 70)
        print(f"Total chunks:       {len(chunks)}")
        print(f"New embeddings:     {embedded_count}")
        print(f"Cached embeddings:  {cached_count}")
        print(f"Total stored:       {len(results)}")
        print(f"Model:              {MODEL_NAME}")
        print(f"Output:             {output_path}")

        return output