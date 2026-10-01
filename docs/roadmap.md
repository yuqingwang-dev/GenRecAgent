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

The time-safe sequence dataset, catalog-valid code trie and collision-aware candidate interface have been implemented and tested. A trained sequence model and its retrieval comparison with Dense remain pending.

Current protocol: [m3_protocol.md](m3_protocol.md). Experiment slot: model configuration, training logs, valid-code generation, and full-catalog Recall/NDCG comparison.

## Phase 4 — Dynamic Tool Router

Route among dense retrieval, generative retrieval, filtering, and reranking based on intent, constraints, confidence, and tool cost.

Experiment slot: routing accuracy, invalid calls, and latency breakdown.

## Phase 5 — Planner + Verifier + Re-planning

Replace the deterministic planner with a provider-backed planner, retain typed traces, and evaluate recovery from invalid or incomplete tool results.

Experiment slot: task success, constraint satisfaction, and recovery rate.

## Phase 6 — Agent Post-training

Collect verified trajectories and add supervised fine-tuning only after a stable offline evaluator exists. Reinforcement learning is outside the current project scope.

Experiment slot: training data version, policy changes, and safety checks.

## Phase 7 — Evaluation and Ablation

Run reproducible comparisons across retrieval strategies, routing policies, verifier settings, and re-planning budgets.

Experiment slot: final tables, confidence intervals, and failure analysis.
