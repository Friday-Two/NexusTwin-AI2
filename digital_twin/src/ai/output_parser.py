import re
from dataclasses import dataclass
from typing import Optional, Tuple


SUPPORTED_OUTCOMES = {
    "allocation",
    "disagreement",
    "no_agreement",
    "disconnect",
}


@dataclass(frozen=True)
class ParsedOutcome:
    raw_text: str
    normalized_text: str
    category: str
    assignments: Tuple[Tuple[str, int], ...]
    is_supported: bool


def normalize_output(text: str) -> str:
    """Normalize generated text using the original evaluation rule."""

    if not isinstance(text, str):
        raise TypeError("Model output must be a string.")

    return " ".join(text.strip().split())


def parse_negotiation_output(text: str) -> ParsedOutcome:
    """
    Classify model output using the original notebook's rules.

    Allocation validation is intentionally handled separately.
    """

    normalized = normalize_output(text)

    assignments = tuple(
        (name, int(value))
        for name, value in re.findall(
            r"\b(item[0-9]+)\s*=\s*(\d+)",
            normalized,
        )
    )

    if re.search(r"\bitem\d+\s*=", normalized):
        category = "allocation"
    elif re.fullmatch(r"(?:<disagree>\s*)+", normalized):
        category = "disagreement"
    elif "<no_agreement>" in normalized.split():
        category = "no_agreement"
    elif "<disconnect>" in normalized.split():
        category = "disconnect"
    else:
        category = "other"

    return ParsedOutcome(
        raw_text=text,
        normalized_text=normalized,
        category=category,
        assignments=assignments,
        is_supported=category in SUPPORTED_OUTCOMES,
    )
