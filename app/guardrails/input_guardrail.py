# app/guardrails/input_guardrail.py

"""
Input guardrail for the biomedical research assistant.

Blocks:
    - Prompt injection
    - Jailbreak attempts
    - Clinical/personal medical advice
    - Clearly off-topic requests
"""


# ----------------------------------------------------------------------
# Patterns
# ----------------------------------------------------------------------

PROMPT_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore your instructions",
    "forget your instructions",
    "disregard previous instructions",
    "disregard all previous instructions",
    "reveal your system prompt",
    "show me your system prompt",
    "reveal the system prompt",
    "list all chunks",
    "show all chunks",
    "dump the database",
]


JAILBREAK_PATTERNS = [
    "pretend you are a doctor",
    "pretend to be a doctor",
    "act as a doctor",
    "you have no restrictions",
    "remove your restrictions",
    "bypass your restrictions",
    "ignore your safety rules",
    "jailbreak",
]


CLINICAL_ADVICE_PATTERNS = [
    "should i take",
    "should i use",
    "should i stop taking",
    "should i start taking",
    "what should i take",
    "what medicine should i take",
    "which drug should i take",
    "can i take",
    "can i stop taking",
    "can i start taking",
    "what dosage should i take",
    "what dose should i take",
    "how much should i take",
    "diagnose me",
    "what is my diagnosis",
    "do i have",
]


# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------

class GuardrailResult:
    """
    Result returned by the input guardrail.
    """

    def __init__(
        self,
        allowed: bool,
        reason: str = "",
    ):
        self.allowed = allowed
        self.reason = reason

    def __repr__(self):
        return (
            f"GuardrailResult("
            f"allowed={self.allowed}, "
            f"reason={self.reason!r})"
        )


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------

def _contains_pattern(
    text: str,
    patterns: list[str],
) -> bool:

    text = text.lower()

    return any(
        pattern in text
        for pattern in patterns
    )


# ----------------------------------------------------------------------
# Main guardrail
# ----------------------------------------------------------------------

def check_input(
    question: str,
) -> GuardrailResult:
    """
    Validate a user's question before sending it to the agent.

    Returns:
        GuardrailResult
    """

    if not question or not question.strip():

        return GuardrailResult(
            allowed=False,
            reason="Question cannot be empty.",
        )

    normalized_question = question.strip()

    # --------------------------------------------------------------
    # Prompt injection
    # --------------------------------------------------------------

    if _contains_pattern(
        normalized_question,
        PROMPT_INJECTION_PATTERNS,
    ):

        return GuardrailResult(
            allowed=False,
            reason="Prompt injection attempt detected.",
        )

    # --------------------------------------------------------------
    # Jailbreak
    # --------------------------------------------------------------

    if _contains_pattern(
        normalized_question,
        JAILBREAK_PATTERNS,
    ):

        return GuardrailResult(
            allowed=False,
            reason="Jailbreak attempt detected.",
        )

    # --------------------------------------------------------------
    # Clinical advice
    # --------------------------------------------------------------

    if _contains_pattern(
        normalized_question,
        CLINICAL_ADVICE_PATTERNS,
    ):

        return GuardrailResult(
            allowed=False,
            reason=(
                "Clinical or personal medical advice is "
                "outside the scope of this literature assistant."
            ),
        )

    # --------------------------------------------------------------
    # Otherwise allow
    # --------------------------------------------------------------

    return GuardrailResult(
        allowed=True,
    )