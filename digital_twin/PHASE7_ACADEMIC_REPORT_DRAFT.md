# Personal AI Digital Twin for Privacy-Aware Multi-Agent Negotiation and Verified Task Execution

## 1. Abstract

This project implements a local prototype of a personal AI Digital Twin that negotiates with simulated service agents under user-defined constraints, applies privacy rules, persists accepted terms in SQLite, and independently verifies the persisted outcome. A FLAN-T5-small model was fine-tuned on a subset of the Deal or No Deal negotiation dataset and evaluated on a held-out test split. The system combines learned negotiation-outcome prediction with explicit rule-based policy enforcement and simulated task execution.

## 2. Objectives

- Represent user preferences, hard constraints, and privacy restrictions in a structured profile.
- Simulate negotiation between two service agents and a personal Digital Twin.
- Reject offers that violate hard constraints.
- Avoid disclosing prohibited phone information.
- Persist accepted terms and verify them independently.
- Evaluate a fine-tuned model against a simple majority-target baseline.
- Save reproducible traces and metrics.

## 3. Methodology

### 3.1 Dataset and model

The primary research dataset is Deal or No Deal. The saved processed data contains 10,095 training examples, 1,087 validation examples, and 1,052 test examples. The model is `google/flan-t5-small`, fine-tuned using the project's saved training procedure. The exported checkpoint and tokenizer are stored under `models/flan_t5_small_negotiation/corrected_fp32/`.

### 3.2 Digital Twin and negotiation

The Digital Twin reads a structured profile containing a preferred area, duration, maximum total price, verification requirement, and privacy preferences. Two simulated service agents produce offers. The negotiation component evaluates offers and records the interaction trace.

### 3.3 Policy enforcement

Explicit checks enforce price and other hard constraints. The profile's privacy setting disallows phone sharing. Policy checks are separate from the neural model so that a generated prediction cannot override the hard constraints.

### 3.4 Execution and verification

The accepted agreement is persisted to a SQLite database. A separate verification function reads the stored record and compares it with the expected agreement and profile constraints. A tamper test changed the stored price from ₹5,900 to ₹9,900; the verifier rejected the modified record and reported price mismatch, snapshot mismatch, and budget violation. After restoration, verification passed again.

### 3.5 Evaluation protocol

The saved test split contains 1,052 examples. Metrics include exact-match accuracy, category accuracy, and whitespace-token micro precision, recall, and F1 for the model. The majority-target baseline predicts the most frequent training target for every test example. Baseline token micro-F1 is not available in the saved baseline artifact and is not compared.

## 4. Results

All reported classifier metrics below use the same held-out Deal or No Deal test split (1,052 examples). The hybrid routing threshold was selected on validation data and frozen before test evaluation.

| Model | Category accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| FLAN-T5-small | 84.51% | 45.26% | 47.82% |
| TF-IDF + Logistic Regression | 78.33% | 59.00% | 62.16% |
| Confidence-routed hybrid (threshold 0.55) | 84.70% | 56.81% | 57.96% |

The hybrid achieved the highest aggregate accuracy (84.70%), while TF-IDF + Logistic Regression achieved the highest macro-F1 (59.00%) and balanced accuracy (62.16%). These metrics measure outcome-category classification; they are not allocation-generation exact-match scores.

For FLAN-T5 generation, exact-match accuracy was 14.92% and token micro-F1 was 63.87%. Exact-match accuracy measures exact target agreement, while token micro-F1 measures whitespace-token overlap; neither alone establishes semantic quality.

The test set is imbalanced. FLAN-T5 did not correctly classify the disagreement or disconnect categories, and the hybrid did not correctly classify any of the 10 disconnect examples. Therefore, high aggregate accuracy should not be interpreted as robust performance across every category.

### Prototype demonstration

The recorded simulated workflow negotiated an agreement for Metro Stay at ₹5,900 for two nights, persisted the record, and independently verified it. A separate impossible-budget scenario failed to produce an accepted offer. A privacy regression validation after the source fix reported these checks as passing: agreement reached, final agreement satisfies constraints, phone not disclosed, no private fields in recorded offers, and offers recorded.

The hotel agents and booking execution are simulated; this is not a live reservation. The privacy audit supports claims about the recorded outgoing offers and tested paths, not every possible communication channel.

## 5. Limitations

1. **Simulated services:** Hotel agents and booking execution are simulated. No live hotel reservation is made.
2. **Model class imbalance:** Minority-class performance is weak. In particular, FLAN-T5 did not correctly classify disagreement or disconnect examples, and the hybrid missed all 10 disconnect test examples.
3. **Limited benchmark scope:** Model evaluation uses a held-out Deal or No Deal test split and does not establish generalization to all service-negotiation domains.
4. **Rule-based policy layer:** Constraint and privacy enforcement use explicit software checks, not learned guarantees.
5. **Verification scope:** The verifier checks persisted fields and agreement consistency. It is not a cryptographic proof and cannot guarantee integrity if all trusted records and verification code are compromised.
6. **Token metric interpretation:** Whitespace-token overlap is a lexical metric, not a measure of semantic correctness.
7. **Privacy audit scope:** The regression audit covers tested behavior and recorded outgoing offers. It does not prove that every possible communication channel is free of private information.
8. **Notebook reproducibility:** A successful script run does not prove that an entire notebook executes top-to-bottom. Notebook-level reproducibility requires a separate run if required for submission.

## 6. Reproducibility

Project root: `/kaggle/working/digital_twin`

Main entry point: `python src/main.py`

Key prototype evidence: `experiments/phase7_end_to_end/`

Final model comparison: `experiments/final_results/final_model_comparison.json`

Privacy-fix validation: `experiments/final_results/privacy_fix_validation.json`

FLAN-T5 generation metrics: `experiments/flan_t5_small_negotiation/test_metrics_corrected.json`

TF-IDF baseline metrics: `experiments/majority_target_baseline/metrics.json` (confirm the file's contents before citing it as the source of TF-IDF metrics; the consolidated final comparison is the authoritative comparison file).

Historical Phase 7 manifest: `experiments/phase7_end_to_end/evidence_manifest.json`

Post-fix manifest: `experiments/final_results/post_privacy_fix_manifest.json`

The historical Phase 7 manifest is retained unchanged. The post-fix manifest records current hashes for selected source and result artifacts. Recompute the manifest if any listed artifact changes. For reproducibility, record the environment, package versions, exact command, and random seeds when available.
 
## 7. Research references

The following are relevant source papers or benchmarks for the project. Verify exact bibliographic metadata and citation style against the original publications before submitting the final report.

1. Deal or No Deal? End-to-End Learning for Negotiation Dialogues. Lewis et al., 2017.
2. TravelPlanner: A Benchmark for Real-World Planning with Constraints.
3. Mind2Web: Towards a Generalist Agent for the Web.
4. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents.
5. Personal Large Language Model Agents.
6. Cooperation, Competition, and Maliciousness in Multi-Agent Systems.
7. Automated Privacy Negotiations with Preference Uncertainty.
8. GAIA: A Benchmark for General AI Assistants.
9. Personalized Tool-Augmented Language Models / ETAPP.
10. Auto-SLURP: research on personal-assistant and multi-agent task execution.
11. AgentBench: Evaluating LLMs as Agents.
12. MultiAgentBench: Benchmarking LLM Agents in Multi-Agent Tasks.
13. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs.
14. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.

## 8. Conclusion

The prototype demonstrates a complete simulated workflow combining multi-agent negotiation, explicit privacy and constraint checks, SQLite persistence, independent verification, and tamper detection. The fine-tuned FLAN-T5-small model improves over the saved majority-target baseline on aggregate test metrics, but its poor performance on minority categories limits claims of robust negotiation-outcome prediction. Further work should focus on class-balanced evaluation, stronger adversarial tests, notebook-level reproducibility, and integration with real service APIs.
