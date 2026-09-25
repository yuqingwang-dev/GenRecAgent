# M1: real All_Beauty development-subset baseline

Run date: 2026-09-25 UTC. This is an offline **development subset of real reviews**, not a full-user experiment and not a purchase or online outcome study. Machine: Intel Core i9-14900HX, 31.6 GiB RAM, RTX 4060 Laptop GPU (8 GiB); BGE encoding used CUDA 12.4/PyTorch 2.6.0, and exact NumPy search used four OpenBLAS threads on CPU. Code commit for the final evaluation: `315554b74da605f3e1cdc5d6c5408f4fe7b29b5f`, clean working tree.

## Data and protocol

- Official `McAuley-Lab/Amazon-Reviews-2023` / `All_Beauty`, revision `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`. Review JSONL SHA-256: `36ca7c48440a8247b2dca0174d095238422b506a8a80c114dca722b7715f1c94`; metadata Parquet SHA-256: `14815cf1312a0d847364866e6876c8c73738993469067242870774c372c04387`.
- 701,528 raw review rows and 112,590 metadata rows. After earliest user-item deduplication and the minimum-three rule, 8,151 users were eligible. SHA256(seed=42, user ID) selected 5,000 users independently of hit outcomes: 21,465 interactions total, 11,465 train, 5,000 validation targets, 5,000 test targets. The candidate universe was the **full** 112,590-item metadata catalog, not sampled negatives.
- Validation used each user's train history. Test used train plus the already observed validation item. Popularity counts used train interactions only. Both methods excluded the respective observed history. No validation or test targets were missing from the candidate catalog or repeated in the observed history.
- Item text used only title, category, features and description; no review text or `average_rating` was encoded. Encoder: `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, 384 dimensions, L2-normalized vectors. User vector: normalized mean of observed item vectors. Dense search: exact inner product over the full catalog, NumPy backend; optional FAISS was not used in this run.

## Results

All values are fractions, not percentages. Recall@K is a hit indicator averaged over all 5,000 users. NDCG@K uses `1 / log2(rank + 1)` for the single relevant target at one-based rank. No user was removed for a missed target.

| Split | Method | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | Popularity | 0.0056 | 0.0080 | 0.0120 | 0.00461 | 0.00540 | 0.00638 |
| Validation | Dense | 0.0296 | 0.0364 | 0.0442 | 0.02231 | 0.02449 | 0.02641 |
| Test | Popularity | 0.0026 | 0.0060 | 0.0086 | 0.00191 | 0.00304 | 0.00370 |
| Test | Dense | 0.0242 | 0.0294 | 0.0368 | 0.01745 | 0.01913 | 0.02099 |

Average retrieval latency on this machine was 0.025/0.026 ms for Popularity and 3.985/3.962 ms for Dense on validation/test respectively. Dense latency includes user vector pooling and exact search, excludes BGE loading and item encoding. The 112,590-item GPU index build, including model load and text encoding, took about 219 seconds. The final two-split CPU evaluation took 41 seconds. These timings are not comparable across machines without matching hardware and scope.

## Interpretation and limits

Dense was higher than Popularity at each reported K in this run, but absolute hit rates were low. The sampled users had only 11,465 / 5,000 = 2.293 train interactions on average, while the candidate catalog contained 112,590 products; sparse histories and a large catalog make this a difficult baseline. This is an observation, not a causal explanation. No encoder, pooling or ranking parameters were chosen using test results.

Reviews are observed events, not necessarily purchases. Static full-catalog metadata is an offline visibility assumption, and per-user leave-last-out does not provide cross-user global time separation. Metadata had 94,886 items with unknown price; M1 did not evaluate price-constrained tasks. The full 8,151 eligible-user run, FAISS-specific backend validation, learned Semantic IDs, generative retrieval and agent policy experiments remain open.

Local reproducibility files (Git-ignored): `outputs/m1_all_beauty_5000/metrics.json`, `outputs/m1_all_beauty_5000/run_config.json`, `data/processed/All_Beauty/manifest.json`, and `data/processed/All_Beauty/index/metadata.json`. Commands: `python scripts/prepare_amazon.py --category All_Beauty --download --revision 2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e --max-users 5000 --seed 42`; `python scripts/build_index.py --device cuda --allow-model-download`; `python scripts/evaluate_dense.py --split both --output-dir outputs/m1_all_beauty_5000` (using a fresh output directory).
