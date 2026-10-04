# M3 validation-only failure analysis and code-rank fusion

Run date: 2026-10-04 (Asia/Shanghai). This follow-up uses the existing server-trained Transformer checkpoint and the frozen train-only code-transition baseline. **Only the 8,151-user validation split was read for outcomes; the test split was not evaluated or used to select a weight.** Both routes use the same 112,590-item catalog, beam size 20, maximum 20 predicted codes, collision expansion, train-only popularity ordering, and seen-item filtering. Diagnostic code commit `4ff2f4f`; rank-fusion code commit `d0315d8`, both clean at their full runs.

The input and checkpoint fingerprints are recorded in the ignored `run_config.json` files under `outputs/m3_validation_diagnostics_full/` and `outputs/m3_rank_fusion_validation_full/`. The checkpoint SHA-256 is `dba260171051311a4e214947bfdaec2449e8cabad0ea6e35907076fb08422dd3`; the codebook and M3 input hashes match the frozen M3 manifest. The diagnostic and fusion endpoints reproduced the earlier full-validation Recall@20 exactly.

## Where retrieval fails

Each user's result is partitioned into three mutually exclusive cases: target code absent from the 20 predicted codes; code predicted but target item absent from the top 20 after expansion; or target item hit. The second case can involve code collisions **or** earlier codes filling the item budget; it is not attributed solely to collision ordering.

| Method | Code absent | Code found, item missed | Item hit | Code Recall@20 | Item Recall@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | 8,096 | 28 | 27 | 0.006748 | 0.003312 |
| Code transition | 8,028 | 52 | 71 | 0.015090 | 0.008711 |

The dominant failure is **code ranking**, not invalid decoding or collision expansion. All three cases sum to 8,151 for each method. Among 5,334 validation targets whose full code never appeared as a train-prefix target code, the Transformer retrieved one item and transition retrieved none. Among the 2,817 targets with a seen full code, Transformer hit 26 and transition hit 71. The 5,242 users with one train-history item had Transformer Recall@20 0.003815 versus transition 0.007631; for 1,438 users with two items the figures were 0.001391 versus 0.010431; for 1,471 with at least three items, 0.003399 versus 0.010877. These strata are descriptive, not causal evidence.

## Fixed validation-only rank-fusion check

To test a cheap backoff without retraining or mixing incomparable raw model scores, the two 20-code lists were fused by weighted reciprocal rank: each code gets `(1-w)/(20+transition_rank) + w/(20+transformer_rank)` where absent-list contributions are zero. The top 20 fused codes use the unchanged item expander. Predeclared Transformer weights were 0.1, 0.25, 0.5 and 0.75; exact 0 and 1 endpoints retain the original lists. This is a validation experiment, not a new trained model.

| Transformer weight | Item hits / 8,151 | Recall@20 | NDCG@20 | Code hits / 8,151 |
| ---: | ---: | ---: | ---: | ---: |
| 0 (transition) | 71 | 0.008711 | 0.003306 | 123 |
| 0.1 | 72 | 0.008833 | 0.003330 | 123 |
| 0.25 | 70 | 0.008588 | 0.003374 | 123 |
| 0.5 | 43 | 0.005275 | 0.002066 | 103 |
| 0.75 | 26 | 0.003190 | 0.001485 | 55 |
| 1 (Transformer) | 27 | 0.003312 | 0.001193 | 55 |

The union of the two original 20-code lists contained the correct code for only **151/8,151 users (1.85%)**, an oracle upper bound on code coverage from these lists. Weight 0.1 adds just one item hit over transition and does not increase code hits; this is too small to claim a reliable gain, and larger weights regress. No weight was promoted to a test run. Keep transition as the generative-route reference and Dense as the strongest measured retrieval route.

The next substantive modeling experiment should address sparse target-code supervision and code generalization on validation, not merely reorder colliding items. Candidates include a stronger train-only prior or a richer interaction dataset; any new method needs a frozen protocol and a genuinely fresh final holdout before making a generalization claim, because the original test results have already been observed. The current fusion code is retained as a reproducible negative ablation.

Evidence: `outputs/m3_validation_diagnostics_full/{metrics.json,run_config.json}` and `outputs/m3_rank_fusion_validation_full/{metrics.json,run_config.json}`. Commands: `python scripts/diagnose_m3_validation.py --device cuda --output-dir outputs/m3_validation_diagnostics_full` and `python scripts/evaluate_m3_rank_fusion_validation.py --device cuda --output-dir outputs/m3_rank_fusion_validation_full`. Both scripts require the full checkpoint and reject nonempty output directories.
