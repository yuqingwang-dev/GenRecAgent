# M3 sequence data and candidate protocol (model not trained)

Historical foundation protocol: the pending-model statements below predate the [completed original K-Means Transformer](results_m3_transformer.md). On 2026-10-09, [new learned RQ-VAE sample preparation](m3_rqvae_stable_plan.md) completed in a separate directory with the same item/user protocol; new full training still requires separate approval. Neither update changes the original mapping/checkpoint or the historical records below.

The completed M3 foundation is a typed data boundary, catalog-valid code decoder and collision-aware candidate adapter. It is **not** a trained generative recommender. The research direction of predicting an item's Semantic ID from a user's prior item IDs follows [TIGER](https://arxiv.org/abs/2305.05065); this repository will measure its own results only after training.

## Frozen inputs and prepared examples

- Data: official Amazon Reviews 2023 / All_Beauty, 8,151 eligible users and the full 112,590-item static metadata catalog from M1. Tokenizer: the separately labeled PCA-whitened residual K-Means baseline in `outputs/m2_rqkmeans_full/item_semantic_ids.json`, with 97,231 valid three-token codes. A collided code maps to **all** catalog items in its group.
- Command: `python scripts/prepare_m3_sequences.py`; settings in `configs/generative.yaml`. Output is Git-ignored under `data/processed/All_Beauty_m3/` with `train.jsonl`, `validation.jsonl`, `test.jsonl` and a provenance `manifest.json` containing source hashes and a clean code commit.
- Training examples: for each user with ordered train items `[A, B, C]`, create `A → B` and `[A, B] → C`; never use that user's validation or test target as a training label. History is capped at the most recent 20 items. Real run: **10,969** training prefixes. Of 8,151 users, **5,242** have only one train item and contribute no train prefix, but remain in both evaluation splits.
- Validation: train history predicts the held-out validation item, **8,151** examples. Test: train history plus the already observed validation item predicts the test item, **8,151** examples. A full-file check confirmed all three split boundaries. Different users may still share an item; this does not turn one user's later interaction into that user's training label. Per-user leave-last-out does not enforce a global timestamp cutoff across users.
- Diagnostic only: 34.56% of validation target *codes* and 31.33% of test target codes appeared among the train-prefix target codes. This suggests limited supervision for unseen full code sequences. It is not a generated recommendation metric and must not be used to remove or tune test examples.

## Valid-code decoding contract

`SemanticCodeTrie` indexes only complete codes present in the frozen catalog. A `NextTokenScorer` supplies log-probabilities over the complete position-specific token vocabulary for the next token given history codes and the already generated prefix. Beam search expands only trie-approved next tokens and accumulates log-probabilities; it cannot return a code absent from the catalog. Empty or non-finite valid proposals yield no code and an explicit candidate shortfall. A fitted count scorer has full-split results; a causal Transformer scorer has only a CPU smoke checkpoint and batched interface tests.

## Collision-aware item retrieval

`GenerativeRetriever` expands each generated code through `SemanticCodebook.lookup`. For a collided code it retains every mapped item, orders that group by **train-interaction popularity** with item ID as the stable tie-breaker, filters items already observed in the request history, deduplicates and returns `item_id`, code log-probability, one-based rank and `source=generative`. Diagnostics include predicted/colliding/invalid/duplicate codes, expanded items, seen-item exclusions, duplicate items and candidate shortfall. The held-out target is not an input to this adapter.

This train-only group ordering keeps the future generative-only route distinct from Dense. Any later reranking by a dense user vector must be reported as a separate hybrid, and fusion should use rank-based or validation-calibrated scores rather than adding raw code log-probabilities to cosine similarity. The local full-catalog uniform-scorer smoke test returned 20 valid unseen items from 20 valid codes; **it is only an interface check**.

## Next acceptance gate

The simple train-only code-transition baseline has been trained and evaluated, with results in [results_m3_transition.md](results_m3_transition.md). The small causal Transformer adapter and a shared evaluator are implemented and CPU-smoke-tested, but full training is pending; [m3_transformer_plan.md](m3_transformer_plan.md) records the proposed budget and acceptance checks. Select checkpoint, beam width and any collision/fusion settings on validation only. Then compare Popularity, Dense, trained generative-only and hybrid retrieval on the same 8,151-user split and full item catalog using Recall/NDCG@5/10/20; additionally report invalid/duplicate/insufficient candidates, collision expansion and latency. Preserve degraded results without claiming online CTR or revenue impact.
