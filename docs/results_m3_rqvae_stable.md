# M3 with stabilized learned RQ-VAE codes: approved full training and evaluation

Run date: 2026-10-09 (Asia/Shanghai). The user explicitly approved “批准 M3 新编码训练及评测”. **One fresh 20-epoch training run, both complete retrieval splits, independent auditing and checksum return are complete.** This closes the approved experiment, not the broader recommendation-quality gap. Old K-Means M3 data/model/results and delivery defaults remain unchanged.

## Frozen experiment and provenance

- Clean training/evaluation source: `3f7f769ea02246284e268da18bcacf1fb25a501c`, isolated server checkout `/mnt/wangyuqing/genrec_upload/m3_rqvae_3f7f769/GenRecAgent`. The existing dedicated Python 3.11.16 / Torch `2.7.1+cu118` interpreter was reused; RTX 3090 had 22.65 GiB free before launch. No driver/package change, pretrained model download, paid API or other user's job termination.
- New [accepted M2 neural RQ-VAE](results_m2.md) mapping, 112,590 static-catalog products and 93,350 valid three-token codes. New samples retain the exact old M3 user/item/history/target protocol: 10,969 train prefixes, 8,151 validation and 8,151 test examples, max history 20. [Preparation and fixed fingerprints](m3_rqvae_stable_plan.md) preserve all-row integrity checks and the prior approval proposal.
- Same original Transformer geometry and optimization: 473,731 parameters; width 128, 2 layers, 4 heads, feedforward 256, dropout 0.1; vocabulary 768 code tokens + 3 markers; max positions 83. Fresh seed 42 initialization, AdamW LR 0.0005, batch 128, gradient clip 1, 20 epochs / 1,720 train steps. Only next-item three-token labels contribute to loss. No old-checkpoint resume or test rows read by training.
- Minimum full-validation target NLL selects the saved checkpoint; fixed beam 20, at most 20 generated codes and K=5/10/20. Collision expansion uses training-interaction popularity and stable item ID, seen filtering and deduplication. No fusion, extra run, test-selected epoch/beam or other search was performed.

Training command: `python scripts/train_semantic_transformer.py --config configs/generative_rqvae_stable.yaml --approved`. Actual runtime **37.999 seconds** before saving / **38.01 seconds** including saving/logging, exit 0. All 20 epochs completed; selected **epoch 6**, validation target NLL **5.42207651567699**, target-token accuracy **0.047642416063468696**. Train NLL fell from 6.33757 to 3.00531; validation NLL rose from its minimum to 6.01831 at epoch 20, consistent with continued overfitting rather than a resolved modeling problem.

## Full retrieval results

Every split has one target per user and denominator **8,151**. Values below are fractions, not percentages. M1 and old M3 results are preserved references on the same catalog/user split, not newly rerun baselines. The two Transformer runs share the exact 20-item M3 contexts; other representations/hardware retain their documented baseline scopes.

| Split | Method | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | Popularity (M1) | 0.004294 | 0.006380 | 0.010183 | 0.003468 | 0.004126 | 0.005041 |
| Validation | Dense (M1) | 0.031039 | 0.038891 | 0.046865 | 0.023222 | 0.025732 | 0.027733 |
| Validation | Transformer / RQ-KMeans (old) | 0.000859 | 0.001840 | 0.003312 | 0.000511 | 0.000827 | 0.001193 |
| Validation | Transformer / learned RQ-VAE (new) | 0.001472 | 0.002454 | 0.004049 | 0.000935 | 0.001250 | 0.001637 |
| Test | Popularity (M1) | 0.002086 | 0.004294 | 0.006625 | 0.001506 | 0.002197 | 0.002784 |
| Test | Dense (M1) | 0.024046 | 0.029690 | 0.036560 | 0.017661 | 0.019504 | 0.021247 |
| Test | Transformer / RQ-KMeans (old) | 0.001227 | 0.001840 | 0.002454 | 0.000769 | 0.000962 | 0.001114 |
| Test | Transformer / learned RQ-VAE (new) | 0.000736 | 0.001350 | 0.002699 | 0.000516 | 0.000718 | 0.001056 |

Actual new validation hits at K=5/10/20: **12 / 20 / 33**, versus old **7 / 15 / 27**. New test hits: **6 / 11 / 22**, versus old **10 / 15 / 20**. Thus test Recall@20 increases by only **2 net hits**, while Recall@5/10 and all test NDCGs decrease. Exact test Recall@20 is **0.002699055330634278**, NDCG@20 **0.0010557533167855472**. There is no overall/robust-gain or superiority claim; relevance remains below Popularity, the old code-transition baseline and Dense. Cross-tokenizer target NLL/accuracy are diagnostics of different code targets, not interchangeable relevance scores.

| New diagnostic total | Validation | Test |
| --- | ---: | ---: |
| Generated valid codes | 163,020 | 163,020 |
| Generated collision-group codes | 76,341 | 82,068 |
| Expanded catalog items | 451,685 | 496,751 |
| Seen items excluded | 450 | 480 |
| Invalid / duplicate codes / duplicate items / candidate shortfall | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

All users received 20 valid unseen item candidates. Post-evaluation target-code train coverage is 35.04% validation / 32.36% test, diagnostic only, never used to remove users or select settings. Mean raw retrieval latency on lab95 is **16.2522 / 16.3105 ms** (validation/test). The frozen evaluator's metrics include lightweight in-memory trace capture and report **16.5215 / 16.5791 ms**. JSONL writing occurs after each full split, outside per-user latency. Complete evaluation runtime is **273.682 seconds** (~4 min 34 sec), exit 0. The old evaluation used another GPU; these timings do not prove a like-for-like speed gain.

## Independent checks and local return

Before retrieval, **20 independent training checks passed**: reload geometry/vocabulary, finite checkpoint tensors and all losses, actual updates to 28 parameter tensors versus fresh seeded initialization, frozen source/input/config hashes, complete epochs, validation-only selection, causality and no evaluation mutation. Manually reconstructed teacher-forced inputs and target-only labels for all 8,151 validation rows / 24,453 supervised tokens reproduce the selected NLL and accuracy exactly. The old server checkpoint SHA remains unchanged. A real old-checkpoint/new-map evaluation request is rejected before inference/output creation, confirming mapping isolation.

The unchanged frozen evaluator ran once for both full splits through a runtime-only trace wrapper, with no source/config/ranking change. Only observed history and top_k enter retrieval; held-out targets are audit labels. The actual per-user candidates, predicted codes, scores, ranks, diagnostics and raw timings were saved. CPU-only independent grading rebuilt collision groups, train-only ordering and seen filtering for **all 16,302 executions / 326,040 generated codes**, then recalculated every Recall/NDCG and diagnostic total. All agree (metric tolerance 1e-12). After return, local CPU repeated the complete regrading, again passing. Full server software regression: **190 passed in 7.27 seconds**, exit 0.

Local fresh paths:

- Model: `outputs/m3_transformer_rqvae_stable_full/` (`transformer.pt`, vocabulary, training log, metrics and run config).
- Both-split evaluation: `outputs/m3_transformer_rqvae_stable_evaluation/`.
- Actual candidates: `outputs/m3_retrieval_traces/{validation,test}.jsonl`.
- Audits: `outputs/m3_training_verification.json`, `outputs/m3_retrieval_verification.json`, `outputs/m3_mapping_isolation_verification.json` and `outputs/m3_evaluation_execution.json`.
- Return/archive/manifest/executable audit methods: `outputs/m3_rqvae_stable_server/`; `local_receipt_verification.json` and `local_regrading.json` record local acceptance. All **31** regular entries pass safe path/type/size checks, archive and per-file hashes, exact local source-input checks and preserved old model/map/sample fingerprints. No old artifact was overwritten.

Returned archive SHA256: `83d9b63f35daa80906aaac2327c58b52f6c020a260c3ff919c66aa6a122996e0`. New checkpoint SHA256: `5978ffcdc3a9489b7511886d733bf4567766c4281877f5c895d74f4fbecb85ec`. All trace/model/evaluation/source fingerprints are retained in the run configs, audit JSONs and receipt manifest. The code/config can reproduce a run only with **fresh** output paths; do not relaunch this completed job or overwrite existing evaluation evidence.

## Acceptance and next boundary

The learned M2-to-M3 chain is now actually trained/evaluated, not merely a Demo or fixed clustering proxy. Structural correctness, complete evaluation and artifact delivery pass. **Weak next-item relevance remains unresolved.** One seed, few hits, already observed historical test set, sparse prefix supervision, per-user rather than global-time splits and static catalog assumptions limit interpretation. This is not a fresh independent holdout, online CTR/revenue evidence or a robust improvement claim.

Keep Dense as the main retrieval reference and the old M3/delivery defaults unchanged. The new model is separately accessible through its explicit config/checkpoint; never mix token maps/checkpoints. Final report/demo consolidation can use this honest learned-tokenizer/generator evidence alongside the already completed tool/SFT chain. Any further model search, larger training or changed held-out protocol requires a new scoped proposal/approval, not reuse of this completed run's authorization.
