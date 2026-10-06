# app/retrieval/retriever.py

from pathlib import Path
import sys

from sentence_transformers import SentenceTransformer

from app.retrieval.vector_store import BiomedicalVectorStore


# ======================================================================
# CONFIGURATION
# ======================================================================

MODEL_NAME = "BAAI/bge-small-en-v1.5"

DEFAULT_VECTOR_STORE = Path(
    "data/vector_store/chroma"
)


# ======================================================================
# RETRIEVER
# ======================================================================

class BiomedicalRetriever:

    def __init__(
        self,
        persist_directory: Path = DEFAULT_VECTOR_STORE,
        top_k: int = 5,
    ):

        self.top_k = top_k
        self.persist_directory = persist_directory

        self._model = None
        self._vector_store = None

    # ==================================================================
    # MODEL
    # ==================================================================

    @property
    def model(self) -> SentenceTransformer:

        if self._model is None:

            print(
                f"Loading embedding model: {MODEL_NAME}",
                file=sys.stderr,
            )

            self._model = SentenceTransformer(
                MODEL_NAME
            )

            print(
                "Embedding model loaded.",
                file=sys.stderr,
            )

        return self._model

    # ==================================================================
    # VECTOR STORE
    # ==================================================================

    @property
    def vector_store(self) -> BiomedicalVectorStore:

        if self._vector_store is None:

            self._vector_store = BiomedicalVectorStore(
                persist_directory=self.persist_directory
            )

        return self._vector_store

    # ==================================================================
    # RETRIEVE
    # ==================================================================

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict]:

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        k = (
            top_k
            if top_k is not None
            else self.top_k
        )

        if k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        # --------------------------------------------------------------
        # Query embedding
        # --------------------------------------------------------------

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        # --------------------------------------------------------------
        # Chroma search
        # --------------------------------------------------------------

        results = self.vector_store.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=k,
        )

        ids = results.get(
            "ids",
            [[]],
        )

        if not ids or not ids[0]:

            return []

        documents = results.get(
            "documents",
            [[]],
        )

        metadatas = results.get(
            "metadatas",
            [[]],
        )

        distances = results.get(
            "distances",
            [[]],
        )

        retrieved_chunks = []

        # --------------------------------------------------------------
        # Convert Chroma results
        # --------------------------------------------------------------

        for i, chunk_id in enumerate(
            ids[0]
        ):

            metadata = {}

            if (
                metadatas
                and metadatas[0]
                and i < len(metadatas[0])
                and metadatas[0][i]
            ):

                metadata = metadatas[0][i]

            source_text = ""

            if (
                documents
                and documents[0]
                and i < len(documents[0])
            ):

                source_text = (
                    documents[0][i]
                    or ""
                )

            distance = None

            if (
                distances
                and distances[0]
                and i < len(distances[0])
            ):

                distance = distances[0][i]

            retrieved_chunks.append(
                {
                    "chunk_id": chunk_id,

                    "source_text": source_text,

                    "document_id": metadata.get(
                        "document_id"
                    ),

                    "page_number": metadata.get(
                        "page_number"
                    ),

                    "chunk_index": metadata.get(
                        "chunk_index"
                    ),

                    "content_hash": metadata.get(
                        "content_hash"
                    ),

                    "chunking_strategy": metadata.get(
                        "chunking_strategy"
                    ),

                    "distance": distance,
                }
            )

        return retrieved_chunks

    # ==================================================================
    # SEARCH
    # ==================================================================

    def search_literature(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        return self.retrieve(
            query=query,
            top_k=top_k,
        )

    # ==================================================================
    # DOCUMENT METADATA
    # ==================================================================

    def get_document_metadata(
        self,
        document_id: str,
    ) -> dict:

        if not document_id or not document_id.strip():

            raise ValueError(
                "Document ID cannot be empty."
            )

        results = self.vector_store.collection.get(
            where={
                "document_id": document_id
            }
        )

        metadatas = results.get(
            "metadatas",
            []
        )

        if not metadatas:

            return {
                "document_id": document_id,
                "found": False,
            }

        pages = sorted(
            {
                metadata.get(
                    "page_number"
                )

                for metadata in metadatas

                if (
                    isinstance(metadata, dict)
                    and metadata.get(
                        "page_number"
                    ) is not None
                )
            }
        )

        return {
            "document_id": document_id,
            "found": True,
            "chunk_count": len(metadatas),
            "pages": pages,
        }