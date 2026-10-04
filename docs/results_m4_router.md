# M4 first increment: dense-first offline routing and bounded fallback

Run date: 2026-10-04 (Asia/Shanghai). This is a **history-based offline retrieval router**, not a natural-language or provider-backed agent. It reuses the full M1 catalog/index and train-interaction popularity. The default policy does not auto-select generative retrieval because both measured M3 generative routes underperformed Dense. The code-transition generator is available only through an explicit experimental mode. No new model was trained.

## Policy and safety checks

| Request | Attempt order | Rule |
| --- | --- | --- |
| `auto` or `dense`, at least one indexed history item | Dense → Popularity | Dense is the measured primary route; Popularity is train-only fallback. |
| `auto` or `dense`, no indexed history item | Popularity | Cold start does not fabricate a user embedding. |
| Explicit `generative` | Code transition → Dense → Popularity | The weaker route is opt-in and falls back if it errors or returns invalid/incomplete candidates. |

The router verifies the requested candidate count (capped by eligible catalog size), catalog membership, no previously seen or duplicate items, contiguous ranks, finite scores, matching source labels, and generative diagnostics. Errors are converted into bounded trace attempts rather than an unbounded re-plan. If every route fails or the catalog has no eligible item, `satisfied=false`; it does not claim a successful recommendation. It does **not** interpret text queries or enforce price/category/availability constraints. The Phase 0 mock Planner/Router/Verifier remain separate; this is a real-data retrieval integration seam, not a finished Qwen policy.

## Full validation result

The optimized `auto` policy ran on all **8,151 validation histories** and the same **112,590-item static catalog** as M1. All 8,151 requests selected Dense with one successful attempt; fallback and failure counts were zero. No target was unreachable or already in its observed history. All six ranking metrics matched the frozen M1 Dense validation result exactly, including Recall@20 **0.04686541528646792** and NDCG@20 **0.02773278725191087**. This is a parity/safety result, **not** a recommendation-quality gain.

An additional full-validation run explicitly requested `generative`. **8,148** users received the code-transition route; **3** encountered insufficient generated item candidates and fell back to Dense; final failed requests were **0**. All six ranking metrics exactly matched the original code-transition validation baseline (Recall@20 **0.008710587657956079**), so this run provides evidence that the fallback triggers on real data, not evidence that generation improved quality. The three affected users had no change in aggregate hits; that should not be assumed for a different dataset.

The optimized router's mean measured route latency was **8.36 ms/user**, including policy, user embedding, exact search and result verification, excluding index/data loading. M1 Dense reported 5.21 ms/user for pooling and search alone in a separate run, so those timings are not directly interchangeable. An initial M4 run averaged 14.97 ms/user because it copied the whole 112,590-item catalog set on every request just to count eligibility. Commit `7eae1af` replaced that O(catalog-size) operation with membership checks over the request's seen items; both ignored outputs were preserved, and the optimized run is the reported one.

The optimized default full run used code commit `7eae1afe8bc1fe515d202aa16f5e71685ff9b199`; the explicit-generative full run used `075abe58eea100197cae1b64e158c988459e21fc`. Both were clean. The prepared-history SHA-256 was `b8d1bde5a643fa64b8791ac0cf368eb691bb8eba0bba1a80d03e4acaf128eb6b`; the loader verified the prepared metadata fingerprint against the Dense index. The explicit route also verified the M3 Semantic ID and train-example hashes and fitted its scorer from train prefixes only. Both evaluations read validation targets only. Cold-start and injected tool/candidate failures were checked in unit tests and command-line traces, not claimed as a full real-user fault benchmark. The project's complete suite passed **54 tests** and Ruff.

## Reproduction and scope

Git-ignored evidence: `outputs/m4_router_validation_full_optimized/{metrics.json,run_config.json}` and `outputs/m4_router_generative_validation_full/{metrics.json,run_config.json}`. The earlier latency-debug run remains at `outputs/m4_router_validation_full/`. Run `python scripts/evaluate_m4_router.py --mode auto --output-dir outputs/m4_router_validation_full_optimized` or `--mode generative` with a different fresh output directory. To inspect a trace without model downloads, run `python scripts/run_m4_router.py --history-item B01DNS649G --top-k 3` (Dense), omit `--history-item` (Popularity cold start), or add `--mode generative` (explicit experimental route). The JSON trace records each attempt, decision reason, selected route, fallback flag and candidate provenance.

The next M4 increments require a separate task set and verifier design for query intent and hard constraints. In particular, products with unknown price must not be presented as satisfying a maximum-price request. No Qwen tool-calling, learned reranker, online CTR or business outcome is claimed here.
