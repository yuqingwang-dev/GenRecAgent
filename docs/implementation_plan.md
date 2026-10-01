# Implementation plan and audit (2026-09-25)

## Audit baseline

Repository: `F:/AIProjects/GenRecAgent`, branch `main`, commit `f5baee6` at audit start. Working tree was clean. The existing `phase0-scaffold` tag points to Phase 0. Interpreter: `C:/Users/Sunny/.conda/envs/genrecagent/python.exe`, Python 3.11.16. Existing tests: 5 passed; Ruff passed. Local hardware: Intel Core i9-14900HX (24 cores/32 logical processors), 33,959,694,336 bytes physical RAM (about 31.6 GiB), and NVIDIA GeForce RTX 4060 Laptop GPU with 8188 MiB total VRAM. No external dataset or model was present in this repository at audit time.

| Area | Status at audit | Evidence / gap |
| --- | --- | --- |
| Phase 0 agent state, mock tools, bounded loop | A: implemented and exercised | `tests/test_*.py`, 5 passing tests; mock catalog only |
| Recommendation metrics helpers | A: unit checked, not benchmarked | `evaluation/recommendation.py`; single simple test |
| OpenAI-compatible LLM | C: stub | `generate` and `tool_call` raise `NotImplementedError` |
| Dense retrieval | C: stub | `retrieval/dense.py` raises `NotImplementedError` |
| Data preparation and index/evaluation CLIs | C: placeholders | scripts only print Phase 0 messages |
| Semantic ID and generative retrieval | C: placeholders | no learned codebook or sequence model |
| Real All_Beauty data, trained encoder, experiment results | D: missing | no downloaded data or model in the project |
| Qwen tool policy, trajectories, SFT, ablations | D: missing | planned only |

`A` means implemented and run locally, `B` means implemented but not yet verified on its intended input, `C` means mock or placeholder, and `D` means absent. This table describes the state *before* the M1 changes below. The README tracks the current state.

## Milestones

1. **M1 — All_Beauty + Popularity and Dense baselines.** Normalize review and item records using `parent_asin`; record filtering and sampling; chronological leave-last-out; static-catalog text encoding; normalized item and user vectors; exact cosine retrieval, saved index, and train-only popularity; validation/test Recall and NDCG. Unit and small CPU fixture tests precede a separately authorized real data/model run.
2. **M2 — RQ-VAE and Semantic ID.** Train residual quantization on M1 item embeddings, save codebooks and item assignments, measure reconstruction, utilization and collision rates, and test collision resolution.
3. **M3 — Transformer generative retrieval.** Build train-prefix examples, train and validate a constrained Semantic ID generator, then compare popularity, dense, generative, and rank-based fusion under one catalog and split.
4. **M4 — Qwen tool policy and verifier.** Introduce validated tool schemas and provider configuration; run fixed-flow and prompt-policy modes with actual tool traces, hard constraint checks, budgets and bounded recovery.
5. **M5 — Tool trajectories and SFT.** Execute and verify diverse trajectories, split tasks by user/template/constraint, then train an adapter only after its model, dataset size and resource estimate are approved.
6. **M6 — Unified evaluation and evidence.** Run controlled ablations, record failures and costs, provide a CLI trace demo, and connect each prospective resume claim to reproducible artifacts.

## M1 acceptance and pause point

The M1 implementation can be checked on deterministic fixture rows without downloading a model. At verified revision `2b6d039`, the official All_Beauty review file is raw JSONL (about 327 MB) and metadata is Parquet (about 60 MB); a review Parquet path in an older example returns 404. Real metrics require those files and a sentence encoder with its runtime. The code must not report a fixture number as a real experiment. Before the first real run, confirm the download volume, model choice, local cache paths, expected execution scale and whether to run a small real subset. Full M1 acceptance requires a logged real-data run with both baselines under the same protocol.

2026-09-25 update: the user approved the download/runtime work. The 5,000-user real development subset ran on the complete 112,590-item catalog; [results_m1.md](results_m1.md) records it. The later [full eligible-user run](results_m1_full.md) covered all 8,151 users passing the minimum-three rule, with the same catalog and model. Both runs report Popularity and Dense on validation and test. M1's real-data baseline checkpoint is satisfied. FAISS-specific parity remains unmeasured; NumPy exact search was the measured backend.

2026-10-01 update: M2 RQ-VAE and collision-aware mapping code is implemented and tested on a 128-item CPU smoke run. Full-catalog training and its validity checks await approval; see [m2_training_plan.md](m2_training_plan.md). No generative retrieval or recommendation gain is claimed from the smoke result.

Later 2026-10-01 update: the laboratory full-catalog RQ-VAE run was received and independently verified, but its three codebooks collapsed severely. A separately labeled PCA-whitened residual MiniBatch K-Means baseline was implemented and run on the same full catalog and item split. [results_m2.md](results_m2.md) preserves both outcomes and the remaining collision limitations. M3 may use the K-Means codes only with explicit collision expansion/ranking and without claiming RQ-VAE success.

M3 foundation update: 10,969 train-only user-history prefixes, 8,151 validation examples and 8,151 test examples are prepared from the full eligible-user set. Catalog-valid code decoding and target-independent collision expansion are implemented and tested, but the scorer is not backed by a trained Transformer. [m3_protocol.md](m3_protocol.md) defines the training/evaluation boundary and next acceptance gate.

M3 next-step update: the train-only code-transition scorer was fitted and evaluated on all validation/test users with fixed settings; its test Recall@20 was 0.007852 versus the M1 Dense 0.036560. A 473,731-parameter causal Transformer training/adapter and shared evaluator are implemented and passed small CPU smoke tests; full training has not run. See [results_m3_transition.md](results_m3_transition.md) and [m3_transformer_plan.md](m3_transformer_plan.md).

## Protocol decisions to preserve

- Reviews are observed interactions, not proof of purchase. `verified_purchase` remains a data field, not a prerequisite for inclusion.
- Use `parent_asin` for both reviews and metadata. Drop interactions with no matching metadata. Keep the earliest review per `(user_id, parent_asin)`; ties use source row order. Sort each user by `(timestamp, source row index)`; equal timestamps are a deterministic but not observable within-time ordering.
- Retain items with unknown price, mark it missing, and do not claim a price constraint is satisfied from that value. Do not use metadata rating statistics as a learned rank feature in M1.
- Require at least three distinct valid interactions per user; use all but last two for train, penultimate for validation, last for test. Validation history is train only. Test history is train plus the already observed validation item. Popularity and any learned model use train interactions only.
- Candidate universe is the retained static metadata catalog, including products with no train interaction. This is an offline static catalog visibility assumption; it is not a historical availability guarantee. No target is removed based on model hits. Targets absent from the chosen catalog remain denominator failures and are counted explicitly.
- Per-user leave-last-out does not enforce global time separation across users. A train event from one user can be later than another user's test event. This limitation must accompany all reported results.
- A development subset samples users with a fixed seed independently of target hit outcomes. Any item cap must be selected independently of validation/test targets and must report unreachable targets. Full catalog evaluation is the default when no cap is supplied.

Sources: [Amazon Reviews 2023 dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023), its [All_Beauty review JSONL](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw/review_categories), and [metadata Parquet](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw_meta_All_Beauty). Direct file loading avoids executing the repository's remote dataset loader script.
