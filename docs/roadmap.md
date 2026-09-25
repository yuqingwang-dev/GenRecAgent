# Roadmap

## Phase 0 — Modern Agent Scaffold

Completed: typed state, mock tools, planner/router/executor/verifier loop, tests, and local demo.

Experiment slot: add trace screenshots or measured smoke-test output here when useful.

## Phase 1 — Dataset + Dense Retrieval Baseline

Completed on a real 5,000-user All_Beauty development subset with the full item catalog: chronological preprocessing, BGE embeddings, an exact index, and train-only Popularity versus Dense Recall@K/NDCG@K. The full eligible-user population and FAISS backend remain unmeasured.

Experiment record: [results_m1.md](results_m1.md) includes dataset/model revisions, split policy, baseline metrics and limits.

## Phase 2 — Semantic ID

Implement item semantic tokenization with RQ-KMeans or RQ-VAE, codebook persistence, collision analysis, and compatibility with the retrieval tool interface.

Experiment slot: codebook settings, token statistics, and ablations.

## Phase 3 — Generative Retrieval

Add a sequence model that generates semantic IDs and compare its retrieval quality and latency with the dense baseline.

Experiment slot: training configuration, generation constraints, and benchmark results.

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
