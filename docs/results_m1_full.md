# M1: All_Beauty full eligible-user baseline

Run date: 2026-09-25 UTC. This is an offline next-item experiment on **all users with at least three valid distinct All_Beauty review interactions** under the documented preprocessing rules. It is not an all-reviewer benchmark: users with fewer than three interactions are excluded by protocol. It is not a purchase, live-catalog or online business outcome study.

## Provenance and setup

- Official `McAuley-Lab/Amazon-Reviews-2023` / `All_Beauty`, dataset revision `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e`; review JSONL SHA-256 `36ca7c48440a8247b2dca0174d095238422b506a8a80c114dca722b7715f1c94`, metadata Parquet SHA-256 `14815cf1312a0d847364866e6876c8c73738993469067242870774c372c04387`.
- 701,528 raw review rows; 112,590 metadata products; 7,599 duplicate user-item review rows discarded. All 8,151 users with at least three distinct valid interactions were retained: **35,422 interactions**, 19,120 train, 8,151 validation targets and 8,151 test targets. Seed 42 was recorded, though full mode does not sample users.
- Full 112,590-item static metadata catalog as candidate universe. Items with no train interaction remain eligible. Targets absent from the catalog: 0; targets already in the observed history: 0. Validation history is train; test history is train plus validation. Popularity counts use only train interactions. No sample was removed based on model output.
- Encoder `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; title, category, features and description only. No review text or aggregate rating is encoded. 384-dimensional L2-normalized item vectors; user vector is the normalized mean of observed item vectors. Exact NumPy inner-product retrieval; no sampled negatives or approximate search. The index was built for the same full product catalog during the 5,000-user development run and reused here because item embeddings do not depend on sampled users.
- Evaluation code commit `42e1b19863127851353e8d9461ed0ab55748e5fe`, clean working tree. Hardware: Intel Core i9-14900HX, 31.6 GiB RAM, RTX 4060 Laptop GPU (8 GiB). Item encoding used PyTorch `2.6.0+cu124` and Sentence Transformers `5.7.0`; evaluation used NumPy `2.4.6` on CPU with four OpenBLAS threads.

## Results

All values are fractions. With one target per user, Recall@K is the fraction whose target appears in top K; NDCG@K is `1 / log2(rank + 1)` at one-based rank, or zero when absent. The denominator is all 8,151 eligible users for each split.

| Split | Method | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | Popularity | 0.004294 | 0.006380 | 0.010183 | 0.003468 | 0.004126 | 0.005041 |
| Validation | Dense | 0.031039 | 0.038891 | 0.046865 | 0.023222 | 0.025732 | 0.027733 |
| Test | Popularity | 0.002086 | 0.004294 | 0.006625 | 0.001506 | 0.002197 | 0.002784 |
| Test | Dense | 0.024046 | 0.029690 | 0.036560 | 0.017661 | 0.019504 | 0.021247 |

Average retrieval latency on this machine: validation Popularity 0.032 ms, Dense 5.215 ms; test Popularity 0.032 ms, Dense 4.937 ms. Dense timing includes user-vector pooling and exact search, excluding model loading and item encoding. The two-split evaluation took 84.85 seconds. Do not compare these latencies with runs on other hardware without matching scope.

## Interpretation and limits

Dense was higher than Popularity for every reported K, but test Recall@20 was only 0.03656; most next-review items were missed. Users averaged 19,120 / 8,151 = 2.346 train items, so histories are short relative to the 112,590-item catalog. That is a plausible contributor, not a proven explanation. These are observed review events, not necessarily purchases, and no test result was used to choose parameters.

Per-user leave-last-out does **not** impose a global time cut across users. The full static catalog is an offline visibility assumption, not proof that every item was available at each historical timestamp. 94,886 metadata items lack price; this experiment does not assert price-constraint satisfaction. FAISS-specific parity, learned Semantic IDs, generative retrieval, agent routing and SFT remain untested or unimplemented at this milestone.

Local Git-ignored evidence: `data/processed/All_Beauty_full_optimized/manifest.json`, `data/processed/All_Beauty/index/metadata.json`, and `outputs/m1_all_beauty_full/{metrics.json,run_config.json}`. The full preparation used the cached official files with `--source-kind official_local --revision 2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e` and no `--max-users`; evaluation used `python scripts/evaluate_dense.py --split both --prepared-dir data/processed/All_Beauty_full_optimized --index-dir data/processed/All_Beauty/index --output-dir outputs/m1_all_beauty_full`. A separate 5,000-user development subset is retained in [results_m1.md](results_m1.md); its results must not be presented as this full-eligible-user run.
