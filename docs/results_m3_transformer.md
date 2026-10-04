# M3 causal Transformer: full training and next-item evaluation

Run date: 2026-10-02 (Asia/Shanghai). This is an offline next-review-item experiment on the same 8,151 eligible users and 112,590-item static catalog as M1 and the M3 code-transition baseline. The Transformer was trained on the independent M2 RQ-KMeans Semantic IDs, **not** on the collapsed RQ-VAE codes.

## Frozen protocol and provenance

- Code commit `02efbf6e41adaa718fa527aaf3a3cff47fc353f5`, clean at training. The server training copy was cloned from the matching Git bundle into `/mnt/wangyuqing/genrec_upload/m3_server_upload_02efbf6/GenRecAgent`; the older server project and its M2 outputs were not overwritten.
- SHA-256: M3 `train.jsonl` `43fcf7c637fa13b2a6a77b16673eb7552275619e94945638f56aa94f3fa6d242`; `validation.jsonl` `c599709d510d015187a3ee254be7b7898341d8478e0380148353911b2d0c2f0d`; item Semantic ID mapping `f394e4ce1a39ec7eb078a9dc8141c3438359c52e7fc6a8c40be1e42e05c45c9c`.
- 10,969 train-period prefix examples updated weights; all 8,151 validation examples selected the checkpoint by target-token negative log-likelihood (NLL). Test targets were not read during training or model selection. The model has 473,731 parameters, two causal Transformer layers, width 128, four heads, and a 771-token vocabulary. Only the target item's three Semantic ID tokens contribute to loss.
- Fixed before the full run: AdamW learning rate 0.0005, batch 128, 20 epochs, seed 42; catalog-valid trie decoding with beam size 20 and at most 20 predicted codes. Code collisions expand to real items ordered by train-interaction popularity and stable item ID, then seen items are excluded. No dense-score fusion or test-set parameter tuning.
- Training hardware: laboratory NVIDIA RTX 3090, Python 3.11, PyTorch `2.7.1+cu118`, CUDA. A project-specific environment was created under the new M3 server directory; the existing Python 3.10 PyTorch environment was not changed. Full validation/test retrieval ran locally on RTX 4060 Laptop GPU with PyTorch `2.6.0+cu124`. A 20-user validation check gave identical Recall/NDCG and candidate diagnostics on both machines before the longer server evaluation was stopped; the full retrieval timing is therefore **local-hardware timing**.

## Training and retrieval results

The 20-epoch training took 115.66 seconds. Best validation target-token NLL was **5.41071 at epoch 5**, with target-token accuracy **0.04241**. Training NLL continued down to 3.02257 by epoch 20 while validation NLL rose to 6.02494, consistent with overfitting. The saved checkpoint is the epoch-5 model, not the last epoch. Its SHA-256 is `dba260171051311a4e214947bfdaec2449e8cabad0ea6e35907076fb08422dd3` and it loaded successfully on the local evaluator.

All retrieval values are fractions; each split has one target per user and denominator 8,151. Baselines are the previously reported full-eligible-user results under the same item universe.

| Split | Method | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | Popularity | 0.004294 | 0.006380 | 0.010183 | 0.003468 | 0.004126 | 0.005041 |
| Validation | Dense | 0.031039 | 0.038891 | 0.046865 | 0.023222 | 0.025732 | 0.027733 |
| Validation | Code transition | 0.003558 | 0.005275 | 0.008711 | 0.001896 | 0.002451 | 0.003306 |
| Validation | Transformer | 0.000859 | 0.001840 | 0.003312 | 0.000511 | 0.000827 | 0.001193 |
| Test | Popularity | 0.002086 | 0.004294 | 0.006625 | 0.001506 | 0.002197 | 0.002784 |
| Test | Dense | 0.024046 | 0.029690 | 0.036560 | 0.017661 | 0.019504 | 0.021247 |
| Test | Code transition | 0.003435 | 0.004662 | 0.007852 | 0.002293 | 0.002675 | 0.003450 |
| Test | Transformer | 0.001227 | 0.001840 | 0.002454 | 0.000769 | 0.000962 | 0.001114 |

The Transformer produced 20 valid codes and enough candidates for every evaluated user: zero invalid or duplicate codes, duplicate items, and candidate shortfalls on both splits. It expanded 464,733 item candidates on validation and 384,361 on test; 274 and 303 seen items, respectively, were filtered. The target full code appeared among train-prefix target codes for 34.56% of validation and 31.33% of test examples. Local mean retrieval latency, including Transformer inference, trie search, collision expansion and seen filtering but excluding loading, was 17.15 ms/user on validation and 17.85 ms/user on test. These times are not directly comparable to CPU-only or different-hardware baselines.

## Interpretation and next decision

This is a valid but **negative model result**: test Recall@20 is below code transition (0.007852), Popularity (0.006625), and Dense (0.036560). The learning curve indicates overfitting; very short user histories, only 10,969 train prefixes from 2,909 users, and sparse full-code supervision are plausible contributors, not established causes. Constraint decoding and collision handling worked correctly, so the main problem is ranking quality rather than invalid outputs. Do not present this checkpoint as an improvement over M1/M3 baselines or as an online recommendation outcome.

For the next experiment, keep Dense as the practical reference and first use validation-only diagnostics/ablations to separate code-token predictability from item ranking. A smaller or more strongly regularized model and a train-only popularity/code-transition backoff are reasonable candidates, but each needs a validation-led protocol and a fresh final holdout; the already observed test results must not drive iterative tuning.

The subsequent validation-only failure breakdown and rank-fusion ablation are documented in [results_m3_validation_diagnostics.md](results_m3_validation_diagnostics.md). They did not establish a reliable improvement.

Git-ignored evidence copied locally: `outputs/m3_transformer_full_server/{transformer.pt,vocabulary.json,training_log.json,metrics.json,run_config.json}`, `outputs/m3_transformer_validation_full_local/{metrics.json,run_config.json}`, and `outputs/m3_transformer_test_full_local/{metrics.json,run_config.json}`. Server checkpoint and training log remain under the M3 copy's `outputs/m3_transformer_full/`. The fixed training command was `python scripts/train_semantic_transformer.py --device cuda --output-dir outputs/m3_transformer_full`; the local evaluation used `scripts/evaluate_semantic_transformer.py --checkpoint outputs/m3_transformer_full_server/transformer.pt --split validation` and then `--split test`, each with `--device cuda` and a fresh output directory. Both full evaluation records are marked `full_evaluation` and reference the same checkpoint hash.
