# app/agent/agent.py

"""
Biomedical Research Assistant Agent.

Architecture:

    User
      ↓
    Input Guardrail
      ↓
    LangChain Agent
      ↓
    Groq LLM
      ↓
    MCP Tools
      ↓
    Retriever
      ↓
    Retrieval Guardrail
      ↓
    Compact Literature Context
      ↓
    Groq LLM
      ↓
    Structured RAG JSON
      ↓
    Citation Guardrail
      ↓
    Grounding Guardrail
      ↓
    Final Answer

IMPORTANT:
    MCP tools are NOT manually wrapped here.

The MCP server is responsible for retrieval guardrails.
The MCP adapter already manages the MCP → LangChain
content/artifact conversion.
"""

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.agent.prompts import SYSTEM_PROMPT

from app.config.settings import (
    GROQ_API_KEY,
    GROQ_MODEL,
    MCP_SERVER_CONFIG,
)

from langfuse.langchain import CallbackHandler


# ======================================================================
# CREATE BIOMEDICAL AGENT
# ======================================================================

async def create_biomedical_agent():
    """
    Create the single biomedical research agent.

    MCP provides:

        search_literature_tool
        retrieve_chunks_tool
        get_document_metadata_tool

    Retrieval guardrails are applied INSIDE the MCP server.

    Do not modify MCP tool.coroutine here.
    """

    print()
    print("=" * 70)
    print("CONNECTING TO MCP SERVER")
    print("=" * 70)

    # ==================================================================
    # MCP CLIENT
    # ==================================================================

    client = MultiServerMCPClient(
        MCP_SERVER_CONFIG
    )

    # ==================================================================
    # LOAD MCP TOOLS
    # ==================================================================

    tools = await client.get_tools()

    print()
    print("=" * 70)
    print("MCP TOOLS LOADED")
    print("=" * 70)

    if not tools:
        raise RuntimeError(
            "No MCP tools were loaded."
        )

    for tool in tools:
        print(f"- {tool.name}")

    # ==================================================================
    # GROQ MODEL
    # ==================================================================

    model = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=GROQ_API_KEY,
        max_tokens=1000,
    )

    print()
    print("=" * 70)
    print("GROQ MODEL READY")
    print("=" * 70)

    print(
        f"Model: {GROQ_MODEL}"
    )

    # ==================================================================
    # CREATE REAL LANGCHAIN AGENT
    # ==================================================================
    langfuse_handler = CallbackHandler()
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    ).with_config(
    {"callbacks": [langfuse_handler]}
)

    print()
    print("=" * 70)
    print("BIOMEDICAL AGENT READY")
    print("=" * 70)

    return agent, client