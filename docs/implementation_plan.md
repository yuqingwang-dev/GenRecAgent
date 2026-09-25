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

The M1 implementation can be checked on deterministic fixture rows without downloading a model. Real All_Beauty metrics require the two official Parquet shards (roughly 139 MB of reviews and 60 MB of metadata, before cache/processed copies), `pyarrow` and a sentence encoder with its runtime. The code must not report a fixture number as a real experiment. Before the first real run, confirm the download volume, model choice, local cache paths, expected execution scale and whether to run a small real subset. Full M1 acceptance requires a logged real-data run with both baselines under the same protocol.

## Protocol decisions to preserve

- Reviews are observed interactions, not proof of purchase. `verified_purchase` remains a data field, not a prerequisite for inclusion.
- Use `parent_asin` for both reviews and metadata. Drop interactions with no matching metadata. Keep the earliest review per `(user_id, parent_asin)`; ties use source row order. Sort each user by `(timestamp, source row index)`; equal timestamps are a deterministic but not observable within-time ordering.
- Retain items with unknown price, mark it missing, and do not claim a price constraint is satisfied from that value. Do not use metadata rating statistics as a learned rank feature in M1.
- Require at least three distinct valid interactions per user; use all but last two for train, penultimate for validation, last for test. Validation history is train only. Test history is train plus the already observed validation item. Popularity and any learned model use train interactions only.
- Candidate universe is the retained static metadata catalog, including products with no train interaction. This is an offline static catalog visibility assumption; it is not a historical availability guarantee. No target is removed based on model hits. Targets absent from the chosen catalog remain denominator failures and are counted explicitly.
- Per-user leave-last-out does not enforce global time separation across users. A train event from one user can be later than another user's test event. This limitation must accompany all reported results.
- A development subset samples users with a fixed seed independently of target hit outcomes. Any item cap must be selected independently of validation/test targets and must report unreachable targets. Full catalog evaluation is the default when no cap is supplied.

Sources: [Amazon Reviews 2023 dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) and its [All_Beauty review Parquet](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw_review_All_Beauty) and [metadata Parquet](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/raw_meta_All_Beauty). The Parquet path avoids executing the repository's remote dataset loader script.
