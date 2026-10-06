
# app/frontend/streamlit_app.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
"""
Streamlit UI for the Biomedical Research Assistant.

Flow:

    Streamlit
        ↓
    Input Guardrail
        ↓
    LangGraph Biomedical Agent
        ↓
    MCP Tools
        ↓
    Chroma Retriever
        ↓
    Grounded Answer
        ↓
    Citation + Confidence
"""

import asyncio
import json

import streamlit as st

from app.agent.agent import create_biomedical_agent
from app.guardrails.input_guardrail import check_input


# ======================================================================
# PAGE CONFIG
# ======================================================================

st.set_page_config(
    page_title="Biomedical Research Assistant",
    page_icon="🧬",
    layout="wide",
)


# ======================================================================
# PAGE TITLE
# ======================================================================

st.title("🧬 Biomedical Research Assistant")

st.markdown(
    """
Ask questions about the biomedical literature available in the
research corpus.

The assistant retrieves evidence from the indexed literature and
provides grounded answers with citations.
"""
)


# ======================================================================
# SESSION STATE
# ======================================================================

if "agent" not in st.session_state:

    st.session_state.agent = None


if "mcp_client" not in st.session_state:

    st.session_state.mcp_client = None


if "messages" not in st.session_state:

    st.session_state.messages = []


# ======================================================================
# CREATE AGENT
# ======================================================================

async def initialize_agent():

    agent, client = await create_biomedical_agent()

    return agent, client


def get_agent():

    if st.session_state.agent is None:

        with st.spinner(
            "Starting biomedical research agent..."
        ):

            agent, client = asyncio.run(
                initialize_agent()
            )

            st.session_state.agent = agent
            st.session_state.mcp_client = client

    return st.session_state.agent


# ======================================================================
# EXTRACT MESSAGE CONTENT
# ======================================================================

def extract_message_content(
    message,
) -> str:

    content = getattr(
        message,
        "content",
        "",
    )

    if isinstance(
        content,
        str,
    ):

        return content.strip()

    if isinstance(
        content,
        list,
    ):

        parts = []

        for item in content:

            if isinstance(
                item,
                str,
            ):

                parts.append(
                    item
                )

            elif isinstance(
                item,
                dict,
            ):

                text = item.get(
                    "text"
                )

                if text:

                    parts.append(
                        str(text)
                    )

        return "\n".join(
            parts
        ).strip()

    return str(
        content
    ).strip()


# ======================================================================
# PARSE JSON RESPONSE
# ======================================================================

def parse_agent_response(
    content: str,
):

    if not content:

        return None

    content = content.strip()

    # --------------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------------

    try:

        return json.loads(
            content
        )

    except json.JSONDecodeError:

        pass

    # --------------------------------------------------------------
    # Markdown JSON fallback
    # --------------------------------------------------------------

    if content.startswith(
        "```"
    ):

        content = (
            content
            .replace(
                "```json",
                "",
            )
            .replace(
                "```",
                "",
            )
            .strip()
        )

        try:

            return json.loads(
                content
            )

        except json.JSONDecodeError:

            pass

    return None


# ======================================================================
# ASK AGENT
# ======================================================================

async def ask_agent(
    agent,
    question: str,
):

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    messages = result.get(
        "messages",
        [],
    )

    if not messages:

        raise RuntimeError(
            "Agent returned no messages."
        )

    last_message = messages[-1]

    return extract_message_content(
        last_message
    )


# ======================================================================
# SIDEBAR
# ======================================================================

with st.sidebar:

    st.header("About")

    st.write(
        """
        This assistant uses:

        • LangGraph agent
        • Groq LLM
        • MCP tools
        • Chroma vector database
        • Biomedical literature retrieval
        • Citation guardrails
        • Grounding guardrails
        • Langfuse observability
        """
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ======================================================================
# DISPLAY PREVIOUS MESSAGES
# ======================================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ======================================================================
# USER INPUT
# ======================================================================

question = st.chat_input(
    "Ask a biomedical literature question..."
)


if question:

    # --------------------------------------------------------------
    # INPUT GUARDRAIL
    # --------------------------------------------------------------

    input_result = check_input(
        question
    )

    if not input_result.allowed:

        st.error(
            f"Input blocked: {input_result.reason}"
        )

        st.stop()

    # --------------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # --------------------------------------------------------------
    # AGENT
    # --------------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching biomedical literature..."
        ):

            try:

                agent = get_agent()

                response = asyncio.run(
                    ask_agent(
                        agent,
                        question,
                    )
                )

                parsed = parse_agent_response(
                    response
                )

                # --------------------------------------------------
                # STRUCTURED RAG RESPONSE
                # --------------------------------------------------

                if isinstance(
                    parsed,
                    dict,
                ):

                    answer = parsed.get(
                        "answer",
                        response,
                    )

                    st.markdown(
                        answer
                    )

                    # ----------------------------------------------
                    # Confidence
                    # ----------------------------------------------

                    confidence = parsed.get(
                        "confidence"
                    )

                    if confidence:

                        st.caption(
                            f"Confidence: {confidence}"
                        )

                    # ----------------------------------------------
                    # Citations
                    # ----------------------------------------------

                    citations = parsed.get(
                        "citations",
                        [],
                    )

                    if citations:

                        with st.expander(
                            "📚 Citations"
                        ):

                            for citation in citations:

                                document_id = citation.get(
                                    "document_id",
                                    "Unknown",
                                )

                                page_number = citation.get(
                                    "page_number",
                                    "Unknown",
                                )

                                chunk_id = citation.get(
                                    "chunk_id",
                                    "Unknown",
                                )

                                st.markdown(
                                    f"""
                                    **Document:** `{document_id}`

                                    **Page:** `{page_number}`

                                    **Chunk:** `{chunk_id}`
                                    """
                                )

                    display_content = answer

                # --------------------------------------------------
                # NORMAL RESPONSE
                # --------------------------------------------------

                else:

                    st.markdown(
                        response
                    )

                    display_content = response

                # --------------------------------------------------
                # SAVE ASSISTANT MESSAGE
                # --------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": display_content,
                    }
                )

            except Exception as exc:

                st.error(
                    f"Agent error: {type(exc).__name__}: {exc}"
                )

