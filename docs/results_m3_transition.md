# M3 train-only code-transition baseline

Run date: 2026-10-01 (local date). This is a real, full-catalog **count-based sequence baseline**, not a trained Transformer. It used the frozen RQ-KMeans item codes from M2, 10,969 train-prefix examples, all 8,151 eligible users for each evaluation split, and the same 112,590-item static catalog as M1.

The scorer counts target-code tokens only in `train.jsonl`. Its first token depends on the last observed item's first token; later tokens depend on the generated prefix, with train-target global distributions as smoothed backoff (`alpha=10`). The catalog trie masks impossible code paths. Each valid code expands to actual items; collisions use train-interaction popularity and stable item ID for within-group order, followed by seen-item filtering. Beam size and maximum predicted codes were both 20, fixed before testing. No raw cosine score was added to code log-probability. Validation and test runs used clean code commit `1fb0c7e` with unchanged configuration.

All values below are fractions. M1 rows are the previously reported full-eligible-user baselines on the same catalog/split; the transition rows are measured here.

| Split | Method | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | Popularity | 0.004294 | 0.006380 | 0.010183 | 0.003468 | 0.004126 | 0.005041 |
| Validation | Dense | 0.031039 | 0.038891 | 0.046865 | 0.023222 | 0.025732 | 0.027733 |
| Validation | Code transition | 0.003558 | 0.005275 | 0.008711 | 0.001896 | 0.002451 | 0.003306 |
| Test | Popularity | 0.002086 | 0.004294 | 0.006625 | 0.001506 | 0.002197 | 0.002784 |
| Test | Dense | 0.024046 | 0.029690 | 0.036560 | 0.017661 | 0.019504 | 0.021247 |
| Test | Code transition | 0.003435 | 0.004662 | 0.007852 | 0.002293 | 0.002675 | 0.003450 |

On validation, 34.56% of target full codes appeared in train-prefix target codes; on test, 31.33%. Both splits had zero invalid or duplicate generated codes. The candidate shortfall totaled 10 slots on validation and 1 slot on test across 8,151 users each; no user was removed. Test retrieval latency averaged 3.69 ms on this machine, including count scoring, trie beam search, collision expansion and history filtering, but excluding input/model loading. M1 Dense/Popularity timing scopes are described in [results_m1_full.md](results_m1_full.md), so latency values should not be used as a direct infrastructure benchmark.

The weak transition result is plausible with sparse histories and a limited train-prefix set, but this run alone does not establish a cause. It remains a fixed reference for the future Transformer. The model did not learn a code from validation/test targets, did not use test hits for parameter selection and makes no online CTR or revenue claim.

Git-ignored evidence: `outputs/m3_transition_validation/{transition_model.json,metrics.json,run_config.json}` and `outputs/m3_transition_test/{transition_model.json,metrics.json,run_config.json}`. Both refer to the M3 example manifest in `data/processed/All_Beauty_m3/`. Commands: `python scripts/evaluate_transition.py --split validation --output-dir outputs/m3_transition_validation` and, without changing settings, `python scripts/evaluate_transition.py --split test --output-dir outputs/m3_transition_test`.
