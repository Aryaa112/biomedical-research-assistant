#app/agent/model.py
import os

from langchain_groq import ChatGroq


MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "llama-3.1-8b-instant",
)


def get_model() -> ChatGroq:
    """
    Create the Groq chat model used by the
    biomedical research agent.

    GROQ_API_KEY must be available in the
    environment.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set."
        )

    return ChatGroq(
        model=MODEL_NAME,
        temperature=0,
    )