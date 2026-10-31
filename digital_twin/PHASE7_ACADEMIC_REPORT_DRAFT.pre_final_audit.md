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

| Metric | FLAN-T5-small | Majority-target baseline |
|---|---:|---:|
| Test examples | 1,052 | 1,052 |
| Exact-match accuracy | 14.9240% | 13.4981% |
| Category accuracy | 84.5057% | 13.4981% |
| Token micro-F1 | 63.8711% | Not available |

Exact-match improvement over the baseline: +1.4259 percentage points.

Category-accuracy improvement over the baseline: +71.0076 percentage points.

The saved test metrics show 801 of 804 allocation examples correctly categorized, but the model did not correctly categorize the disagreement or disconnect examples. The test set is imbalanced, so aggregate accuracy alone does not demonstrate robust performance across all categories.

### Prototype demonstration

The integrated entry point `python src/main.py` completed successfully in the recorded run. It negotiated an agreement for Metro Stay at ₹5,900 for two nights, persisted the record, and independently verified it. The privacy audit passed. An impossible-budget test returned a failure status and no accepted offer.

## 5. Limitations

1. **Simulated services:** The hotel agents and booking execution are simulated. This is not a live hotel-booking integration.
2. **Model class imbalance:** The model performs poorly on disagreement and disconnect classes. Aggregate metrics can obscure these failures.
3. **Limited benchmark scope:** The reported model evaluation uses a held-out Deal or No Deal test split; it does not establish generalization to all service-negotiation domains.
4. **Rule-based policy layer:** Constraint and privacy enforcement are explicit software checks, not learned guarantees.
5. **Verification scope:** The verifier checks stored fields and agreement consistency; it is not a cryptographic proof or a defense against an attacker who can modify all trusted records and verification code.
6. **Token metric interpretation:** Whitespace-token overlap is a lexical metric and should not be interpreted as semantic quality.
7. **Notebook reproducibility:** A successful script run does not by itself prove that the entire original notebook runs top-to-bottom. The notebook must be located and executed separately if that requirement applies.

## 6. Reproducibility

Project root: `/kaggle/working/digital_twin`

Main entry point: `python src/main.py`

Key evidence directory: `experiments/phase7_end_to_end/`

Saved model metrics: `experiments/flan_t5_small_negotiation/test_metrics_corrected.json`

Saved baseline metrics: `experiments/majority_target_baseline/metrics.json`

Evidence manifest: `experiments/phase7_end_to_end/evidence_manifest.json`

The manifest records SHA-256 hashes for key source, model, and experiment files. The report should be regenerated if the underlying metrics or artifacts change.

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
