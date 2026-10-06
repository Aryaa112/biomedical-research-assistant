import re


# Common signs of broken PDF text encoding.
GARBLED_PATTERNS = [
    "‘",
    "’",
    "©",
    "¤",
    "�",
]


def calculate_text_quality(text: str) -> float:
    """
    Returns a score between 0 and 1.

    Higher = more likely to be usable text.
    """

    if not text or len(text.strip()) < 50:
        return 0.0

    score = 1.0

    # Replacement character is a very strong signal.
    replacement_count = text.count("�")
    score -= min(replacement_count / 20, 0.5)

    # Suspicious characters.
    suspicious_count = sum(
        text.count(pattern)
        for pattern in GARBLED_PATTERNS
    )

    score -= min(suspicious_count / max(len(text), 1) * 10, 0.4)

    # Check alphabetic content.
    letters = sum(c.isalpha() for c in text)
    total = max(len(text), 1)

    alphabetic_ratio = letters / total

    if alphabetic_ratio < 0.25:
        score -= 0.3

    # Excessive weird single-character patterns.
    words = text.split()

    if words:
        short_word_ratio = sum(
            1 for word in words
            if len(word) <= 1
        ) / len(words)

        if short_word_ratio > 0.35:
            score -= 0.2

    return max(0.0, min(score, 1.0))


def is_garbled(text: str, threshold: float = 0.70) -> bool:
    return calculate_text_quality(text) < threshold