from typing import Mapping


def build_negotiation_prompt(
    own_input: str,
    partner_input: str,
    dialogue: str,
) -> str:
    """
    Build a prompt using the original FLAN-T5 training format.
    """

    fields = {
        "own_input": own_input,
        "partner_input": partner_input,
        "dialogue": dialogue,
    }

    for field_name, value in fields.items():
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a string."
            )

        if not value.strip():
            raise ValueError(
                f"{field_name} cannot be empty."
            )

    return (
        "Predict the negotiation outcome.\n"
        f"Your item counts and values: {own_input.strip()}\n"
        f"Partner item counts and values: {partner_input.strip()}\n"
        f"Negotiation dialogue: {dialogue.strip()}\n"
        "Output the original target format."
    )
