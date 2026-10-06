# app/config/settings.py

import os

from dotenv import load_dotenv


# ======================================================================
# LOAD ENVIRONMENT VARIABLES
# ======================================================================

load_dotenv()


# ======================================================================
# GROQ CONFIGURATION
# ======================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured. "
        "Please add GROQ_API_KEY to your .env file."
    )


GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b",
)


# ======================================================================
# MCP CONFIGURATION
# ======================================================================

MCP_SERVER_CONFIG = {
    "biomedical": {
        "transport": "stdio",
        "command": "python",
        "args": [
            "-m",
            "app.mcp.server",
        ],
    }
}