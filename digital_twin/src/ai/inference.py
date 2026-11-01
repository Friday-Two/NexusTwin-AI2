import time
from dataclasses import dataclass
from typing import Optional

import torch

from .allocation_validator import validate_allocation
from .model_loader import load_negotiation_model
from .output_parser import parse_negotiation_output
from .prompt_builder import build_negotiation_prompt


@dataclass(frozen=True)
class NegotiationPrediction:
    checkpoint: str
    raw_output: str
    category: str
    is_supported: bool
    allocation_valid: Optional[bool]
    validation_reason: str
    latency_seconds: float
    device: str

    @property
    def is_acceptable(self) -> bool:
        """True only when a valid allocation is available for downstream use."""
        return (
            self.is_supported
            and self.category == "allocation"
            and self.allocation_valid is True
        )


class NegotiationInference:
    """Run FLAN-T5 negotiation outcome inference."""

    def __init__(
        self,
        checkpoint_name: str = "corrected_fp32",
        device: str = "cpu",
        max_input_length: int = 512,
        max_new_tokens: int = 128,
        num_beams: int = 1,
    ):
        if max_input_length <= 0:
            raise ValueError("max_input_length must be positive.")

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")

        if num_beams <= 0:
            raise ValueError("num_beams must be positive.")

        self.checkpoint_name = checkpoint_name
        self.device = torch.device(device)
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        self.tokenizer, self.model = load_negotiation_model(
            checkpoint_name=checkpoint_name,
            device=device,
        )

    def predict(
        self,
        own_input: str,
        partner_input: str,
        dialogue: str,
    ) -> NegotiationPrediction:
        """Generate, parse, and validate a negotiation prediction."""

        prompt = build_negotiation_prompt(
            own_input=own_input,
            partner_input=partner_input,
            dialogue=dialogue,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        start_time = time.perf_counter()

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
            )

        latency = time.perf_counter() - start_time

        raw_output = self.tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        )

        parsed = parse_negotiation_output(raw_output)

        allocation_valid = None
        validation_reason = "not_an_allocation"

        if parsed.category == "allocation":
            validation = validate_allocation(
                target=parsed.normalized_text,
                own_input=own_input,
            )

            allocation_valid = validation.is_valid
            validation_reason = validation.reason

        elif parsed.category == "other":
            validation_reason = "unsupported_output_format"

        else:
            validation_reason = "recognized_non_allocation_outcome"

        return NegotiationPrediction(
            checkpoint=self.checkpoint_name,
            raw_output=raw_output,
            category=parsed.category,
            is_supported=parsed.is_supported,
            allocation_valid=allocation_valid,
            validation_reason=validation_reason,
            latency_seconds=latency,
            device=str(self.device),
        )