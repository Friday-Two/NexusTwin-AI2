from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# digital_twin/src/ai/model_loader.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "flan_t5_small_negotiation"
)

SUPPORTED_CHECKPOINTS = {
    "best_fp32",
    "corrected_fp32",
}


def load_negotiation_model(
    checkpoint_name: str = "corrected_fp32",
    device: str = "cpu",
) -> Tuple[object, torch.nn.Module]:
    """
    Load a fine-tuned FLAN-T5 negotiation model and tokenizer.

    The loader preserves the independently learned shared and lm_head
    matrices. It uses local checkpoint files only and FP32 inference.
    """

    if checkpoint_name not in SUPPORTED_CHECKPOINTS:
        raise ValueError(
            f"Unsupported checkpoint: {checkpoint_name}. "
            f"Choose one of {sorted(SUPPORTED_CHECKPOINTS)}."
        )

    requested_device = torch.device(device)

    if requested_device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported.")

    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")

    checkpoint_dir = MODEL_ROOT / checkpoint_name

    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (checkpoint_dir / filename).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Checkpoint is incomplete: {checkpoint_dir}. "
            f"Missing files: {missing_files}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_dir),
        local_files_only=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(checkpoint_dir),
        dtype=torch.float32,
        local_files_only=True,
    )

    model.to(requested_device)
    model.eval()

    # Verify the expected T5 architecture.
    if model.config.model_type != "t5":
        raise RuntimeError(
            f"Expected a T5 checkpoint, got {model.config.model_type!r}."
        )

    # Verify decoder scaling is consistent with the saved checkpoints.
    if model.config.scale_decoder_outputs is not False:
        raise RuntimeError(
            "Unexpected decoder scaling configuration."
        )

    # Verify independently trained embedding matrices were preserved.
    shared = model.shared.weight
    lm_head = model.lm_head.weight

    if shared.data_ptr() == lm_head.data_ptr():
        raise RuntimeError(
            "Input and output embedding matrices unexpectedly share storage."
        )

    if not torch.isfinite(shared).all().item():
        raise RuntimeError("shared.weight contains non-finite values.")

    if not torch.isfinite(lm_head).all().item():
        raise RuntimeError("lm_head.weight contains non-finite values.")

    if next(model.parameters()).device != requested_device:
        raise RuntimeError("Model was not loaded onto the requested device.")

    if model.training:
        raise RuntimeError("Model must be in evaluation mode.")

    return tokenizer, model
