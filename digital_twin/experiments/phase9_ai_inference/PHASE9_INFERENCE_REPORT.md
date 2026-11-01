# Phase 9 — FLAN-T5 Inference Engineering Report

**Project:** NexusTwin-AI / Personal AI Digital Twin
**Phase:** 9 — FLAN-T5 AI Integration
**Experiment date:** 2 November 2026
**Checkpoint:** `corrected_fp32`
**Execution device:** CPU
**Status:** Inference pipeline implemented, tested, and smoke-tested with the actual checkpoint

---

## 1. Objective

The objective of this stage was to integrate the locally trained FLAN-T5-small negotiation checkpoint into a modular inference pipeline.

The pipeline must:

1. Construct the negotiation prompt.
2. Load the selected local checkpoint.
3. Generate a negotiation outcome.
4. Parse and classify the generated output.
5. Validate allocation outputs against deterministic item-conservation rules.
6. Return a structured prediction result.
7. Preserve evidence without executing external actions.

## 2. Implemented components

Source directory: `digital_twin/src/ai/`

| File | Responsibility |
|---|---|
| `model_loader.py` | Loads and checks local FLAN-T5 checkpoints |
| `prompt_builder.py` | Constructs the original negotiation prompt |
| `output_parser.py` | Normalizes and classifies generated outputs |
| `allocation_validator.py` | Checks allocation structure, nonnegative values, and item conservation |
| `inference.py` | Coordinates generation, parsing, and validation |
| `__init__.py` | Exposes the public AI interfaces |

Public interfaces:

- `NegotiationInference`
- `NegotiationPrediction`

## 3. Inference pipeline

1. Build the prompt using the original training prompt format.
2. Tokenize the prompt with input truncation.
3. Generate output using deterministic decoding (`do_sample=False`).
4. Decode the generated tokens.
5. Parse the output into a supported outcome category.
6. Validate allocations when the predicted category is `allocation`.
7. Return a structured prediction with latency and validation information.

The model is loaded locally with `local_files_only=True`.

Supported checkpoints:

- `best_fp32`
- `corrected_fp32`

The loader checks the checkpoint files, device availability, T5 architecture, FP32 weights, embedding matrix separation, finite weight values, and evaluation mode.

## 4. Prediction safety and response contract

The response includes:

- `category`
- `is_supported`
- `allocation_valid`
- `validation_reason`
- `is_acceptable`
- `latency_seconds`
- `device`

The response contract distinguishes between recognition and acceptability:

- `is_supported` indicates whether the generated output belongs to a recognized outcome category.
- `is_acceptable` is true only when the output is a supported allocation that passes deterministic validation.

A disagreement, no-agreement, or disconnect outcome can be recognized, but it is not an acceptable allocation.

The allocation validator checks:

- Exactly six assignments.
- Expected assignment order.
- Valid input structure.
- Nonnegative allocation values.
- Conservation of the available item counts.

The inference pipeline does not execute bookings, payments, transfers, or other external actions. Model predictions remain proposals; deterministic application logic must retain execution authority.

## 5. Test results

### 5.1 Complete regression test suite

| Test module | Passed |
|---|---:|
| `test_digital_twin.py` — negotiation engine | 20/20 |
| `test_execution_verifier.py` — execution verifier | 8/8 |
| `test_phase9_allocation_validator.py` — allocation validation | 9/9 |
| `test_phase9_inference.py` — inference pipeline | 9/9 |
| **Total** | **46/46** |

**Result: 46 tests passed.**

The tests cover:

- Valid and invalid allocations.
- Item-conservation failures.
- Negative input and allocation values.
- Missing assignments and incorrect assignment order.
- Non-string input handling.
- Valid allocation prediction.
- Invalid allocation rejection.
- Unsupported model output.
- Disagreement, no-agreement, and disconnect outcomes.
- Deterministic generation settings.
- Invalid generation configuration.

The inference pipeline tests use mocked model and tokenizer objects. They verify software behavior without repeatedly loading the actual checkpoint.

### 5.2 Inference acceptance behavior

| Scenario | Supported | Acceptable |
|---|---|---|
| Valid allocation | True | True |
| Invalid allocation | True | False |
| Disagreement | True | False |
| No agreement | True | False |
| Disconnect | True | False |
| Unsupported output | False | False |

**Inference pipeline tests: 9/9 passed.**

### 5.3 Saved test evidence

The complete test output is saved at:

`digital_twin/reports/phase9/test_results.txt`

The saved result confirms:

```text
Ran 46 tests

OK
```

The test command exited with code `0`.

## 6. Actual checkpoint inference

**Checkpoint:** `corrected_fp32`
**Device:** CPU
**Model load time:** 4.3037 seconds
**Inference latency:** 0.4512 seconds

Input:

- Own input: `1 4 4 1 1 2`
- Partner input: `1 4 4 1 1 2`

Generated output:

```text
item0=1 item1=0 item2=1 item0=1 item1=0 item2=1
```

Parsed category: `allocation`
Supported format: `True`
Allocation valid: `False`
Validation reason: `allocation_does_not_conserve_items`
Acceptable: `False`

The predicted allocation failed item-conservation validation. The validator correctly rejected it rather than allowing an invalid allocation to proceed.

For the supplied input, the required totals are:

- item0: 1
- item1: 4
- item2: 1

The generated output does not satisfy these totals.

This actual checkpoint run demonstrates successful model loading, prompt processing, generation, parsing, and deterministic validation. It does not demonstrate a successful allocation or establish model accuracy.

## 7. Checkpoint loading observation

The checkpoint loaded successfully using the local Transformers environment.

A warning was reported because the runtime configuration expects tied T5 weights, while the saved `shared.weight` and `lm_head.weight` tensors contain different values.

Both saved tensors were preserved separately by the loader. The warning was documented rather than suppressed by modifying model configuration or checkpoint weights.

## 8. Experiment evidence

Evidence directory:

`digital_twin/experiments/phase9_ai_inference/`

Evidence files:

- `inference_result.json` — latest actual checkpoint inference result.
- `inference_result.json.phase9-pre-rerun.bak` — preserved previous inference result.
- `run_inference_demo.py` — actual checkpoint inference script.
- `PHASE9_INFERENCE_REPORT.md` — this engineering report.

The JSON record includes:

- Experiment timestamp.
- Checkpoint and execution device.
- Input values and dialogue.
- Model loading time.
- Generated output and parsed category.
- Allocation validation result.
- Inference latency.
- `execution_performed: false`.

The JSON evidence was inspected and its fields matched the terminal output.

## 9. Limitations and pending work

1. The actual checkpoint smoke test demonstrates successful model loading and inference, not high prediction accuracy.
2. The generated allocation was rejected by the deterministic validator.
3. Mocked pipeline tests do not establish real-world model performance.
4. Recognized non-allocation outcomes require deterministic negotiation-state verification.
5. Integration with the existing deterministic negotiation engine remains pending.
6. The model must not directly authorize external service execution.
7. Further evaluation is required to measure performance across all outcome categories.
8. CPU latency measurements are from a smoke test and should not be treated as benchmark results.

## 10. Conclusion

Phase 9 established a modular FLAN-T5 inference pipeline with local checkpoint loading, prompt construction, output parsing, structured prediction results, and deterministic allocation validation.

The complete regression suite passed all 46 tests. The actual `corrected_fp32` checkpoint loaded successfully and generated an output on CPU.

The generated allocation was rejected because it violated item-conservation rules. The stricter acceptance contract also ensures that recognized non-allocation outcomes are not treated as usable allocations.

This demonstrates an important safety property: successful model inference does not automatically make a prediction valid or executable.

The existing deterministic negotiation engine and independent verification layer must remain authoritative for future integration.
