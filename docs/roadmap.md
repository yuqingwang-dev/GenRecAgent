# Roadmap

## Phase 0 — Modern Agent Scaffold

Completed: typed state, mock tools, planner/router/executor/verifier loop, tests, and local demo.

Experiment slot: add trace screenshots or measured smoke-test output here when useful.

## Phase 1 — Dataset + Dense Retrieval Baseline

Completed on both a real 5,000-user All_Beauty development subset and the full set of 8,151 eligible users, each with the full item catalog: chronological preprocessing, BGE embeddings, an exact index, and train-only Popularity versus Dense Recall@K/NDCG@K. The optional FAISS backend remains unmeasured.

Experiment records: [full eligible-user results](results_m1_full.md) and [development subset results](results_m1.md) include dataset/model revisions, split policy, baseline metrics and limits.

## Phase 2 — Semantic ID

RQ-VAE training and collision-aware item mapping were completed on the full catalog, but the learned codebooks collapsed. A separate PCA-whitened residual MiniBatch K-Means baseline was implemented and run on the same full catalog; it produces substantially more distinct codes while retaining all collisions for downstream handling. RQ-VAE refinement remains open.

Original plan and smoke diagnostics: [m2_training_plan.md](m2_training_plan.md). Full-run comparison and limitations: [results_m2.md](results_m2.md).

## Phase 3 — Generative Retrieval

The time-safe sequence dataset, catalog-valid code trie and collision-aware candidate interface were implemented and tested. A train-only code-transition baseline and a small Transformer were both fully trained/evaluated where applicable; both are weaker than Dense on the held-out test split. Validation-only failure analysis found code prediction, rather than invalid decoding, to be the main limitation. A rank-fusion ablation did not establish a reliable gain.

Protocol: [m3_protocol.md](m3_protocol.md). Full results: [transition](results_m3_transition.md), [Transformer](results_m3_transformer.md), and [validation-only diagnosis](results_m3_validation_diagnostics.md). The pre-run plan remains in [m3_transformer_plan.md](m3_transformer_plan.md).

## Phase 4 — Dynamic Tool Router

Three increments are complete: dense-first routing with bounded fallback and Dense parity ([results_m4_router.md](results_m4_router.md)), controlled intents and hard static-price eligibility ([results_m4_constraints.md](results_m4_constraints.md)), and a real local Qwen JSON policy with grounded repair and rule fallback ([results_m4_llm_policy.md](results_m4_llm_policy.md)). Curated paraphrase coverage improved, while raw model reliability remains weak.

Next experiment slot: more independent expressions and verified trajectory collection, retaining metadata eligibility and raw-policy error reporting.

## Phase 5 — Planner + Verifier + Re-planning

Local Qwen can now compile one typed intent/price decision, with one repair and conservative fallback. Broader multi-step plans and recovery from semantic/tool failures still need a separate evaluated task set.

Experiment slot: task success, constraint satisfaction, and recovery rate.

## Phase 6 — Agent Post-training

Collect verified trajectories and add supervised fine-tuning only after a stable offline evaluator exists. Reinforcement learning is outside the current project scope.

Experiment slot: training data version, policy changes, and safety checks.

## Phase 7 — Evaluation and Ablation

Run reproducible comparisons across retrieval strategies, routing policies, verifier settings, and re-planning budgets.

Experiment slot: final tables, confidence intervals, and failure analysis.
