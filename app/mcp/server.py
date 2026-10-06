# app/mcp/server.py

"""
Biomedical Research Assistant MCP Server.

Responsibilities:

    MCP
      ↓
    BiomedicalRetriever
      ↓
    Retrieval Guardrail
      ↓
    Compact JSON result
      ↓
    LangChain Agent

The MCP server owns retrieval-related guardrails.

The agent does NOT modify MCP tool coroutines.
"""

import json
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.retrieval.retriever import BiomedicalRetriever

from app.guardrails.retrieval_guardrail import (
    check_retrieval,
)


# ======================================================================
# MCP SERVER
# ======================================================================

mcp = FastMCP(
    "Biomedical Research Assistant"
)


# ======================================================================
# SHARED RETRIEVER
# ======================================================================

retriever = BiomedicalRetriever()


# ======================================================================
# CONFIGURATION
# ======================================================================

MAX_RETRIEVAL_CHUNKS = 2
MAX_CHARS_PER_CHUNK = 1800


# ======================================================================
# HELPER — JSON SERIALIZATION
# ======================================================================

def _json_response(
    value: Any,
) -> str:
    """
    Convert tool output into deterministic JSON text.

    Returning text across the MCP boundary avoids ambiguity between
    Python structured objects and LangChain tool artifacts.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
    )


# ======================================================================
# GUARDED RETRIEVAL
# ======================================================================

def guarded_retrieval(
    query: str,
    top_k: int = 5,
) -> str:
    """
    Perform semantic retrieval and apply retrieval guardrails.

    Returns JSON text.

    Possible outputs:

        {
            "retrieval_status": "SUCCESS",
            "chunks": [...]
        }

    OR:

        {
            "retrieval_status": "OUT_OF_CORPUS",
            "message": "...",
            "chunks": []
        }
    """

    # --------------------------------------------------------------
    # Validate query
    # --------------------------------------------------------------

    if not query or not query.strip():

        return _json_response(
            {
                "retrieval_status": "ERROR",
                "message": "Query cannot be empty.",
                "chunks": [],
            }
        )

    # --------------------------------------------------------------
    # Validate top_k
    # --------------------------------------------------------------

    if top_k <= 0:

        return _json_response(
            {
                "retrieval_status": "ERROR",
                "message": "top_k must be greater than zero.",
                "chunks": [],
            }
        )

    # --------------------------------------------------------------
    # Protect against unnecessarily large retrieval
    # --------------------------------------------------------------

    top_k = min(
        top_k,
        5,
    )

    # --------------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------------

    try:

        chunks = retriever.retrieve(
            query=query,
            top_k=top_k,
        )

    except Exception as exc:

        print(
            f"Retrieval error: {exc}",
            file=sys.stderr,
        )

        return _json_response(
            {
                "retrieval_status": "ERROR",
                "message": (
                    "The literature retrieval operation failed."
                ),
                "chunks": [],
            }
        )

    # --------------------------------------------------------------
    # Retrieval guardrail
    # --------------------------------------------------------------

    result = check_retrieval(
        chunks=chunks,
        max_chunks=MAX_RETRIEVAL_CHUNKS,
        max_chars_per_chunk=MAX_CHARS_PER_CHUNK,
    )

    # --------------------------------------------------------------
    # OUT OF CORPUS
    # --------------------------------------------------------------

    if not result.allowed:

        return _json_response(
            {
                "retrieval_status": "OUT_OF_CORPUS",
                "message": result.reason,
                "chunks": [],
            }
        )

    # --------------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------------

    return _json_response(
        {
            "retrieval_status": "SUCCESS",
            "message": result.reason,
            "chunks": result.relevant_chunks,
        }
    )


# ======================================================================
# TOOL 1 — SEARCH LITERATURE
# ======================================================================

@mcp.tool()
def search_literature_tool(
    query: str,
    top_k: int = 5,
) -> str:
    """
    Search the biomedical literature corpus.

    Returns compact JSON containing relevant literature chunks.
    """

    return guarded_retrieval(
        query=query,
        top_k=top_k,
    )


# ======================================================================
# TOOL 2 — RETRIEVE CHUNKS
# ======================================================================

@mcp.tool()
def retrieve_chunks_tool(
    query: str,
    top_k: int = 5,
) -> str:
    """
    Retrieve the most relevant biomedical literature chunks.

    Returns compact JSON containing relevant literature chunks.
    """

    return guarded_retrieval(
        query=query,
        top_k=top_k,
    )


# ======================================================================
# TOOL 3 — DOCUMENT METADATA
# ======================================================================

@mcp.tool()
def get_document_metadata_tool(
    document_id: str,
) -> str:
    """
    Retrieve metadata associated with a biomedical document.

    Returns JSON text.
    """

    if not document_id or not document_id.strip():

        return _json_response(
            {
                "found": False,
                "document_id": document_id,
                "message": "Document ID cannot be empty.",
            }
        )

    try:

        metadata = retriever.get_document_metadata(
            document_id=document_id,
        )

        return _json_response(
            metadata
        )

    except Exception as exc:

        print(
            f"Metadata retrieval error: {exc}",
            file=sys.stderr,
        )

        return _json_response(
            {
                "found": False,
                "document_id": document_id,
                "message": (
                    "Unable to retrieve document metadata."
                ),
            }
        )


# ======================================================================
# SERVER ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    mcp.run()