# Interview notes tied to this repository

## Why these M1 baselines?

Popularity uses only users' train interactions and gives a simple collaborative prior. Dense retrieval uses static product text, mean-pooled vectors of the user's observed items, L2 normalization and exact inner product. The shared catalog and seen-item rules make the two methods comparable. `run_dense_demo.py` is a separate query-to-item task; its output is not evidence for next-item recommendation quality.

## What could make the metrics misleading?

Within-user leave-last-out does not create a global time cut. Static full-catalog metadata may expose descriptions of items unavailable at an earlier event time, and `average_rating` is a scraped aggregate that should not be trained into a historical ranker. Duplicate user-item reviews are collapsed to the earliest event. A missing catalog target stays in the denominator. `source_kind` and input hashes distinguish official data from a synthetic fixture.

## How will a weak result be handled?

Keep the full split and candidate universe fixed. Inspect unreachable targets, sparse train histories, metadata text quality and the difference between frequent and tail items. Any encoder, pooler or fusion choice is selected on validation only; test is held for the final comparison. Do not infer online CTR, conversion or revenue from these offline metrics.

The first real development-subset test result was Recall@20 = 0.0086 for Popularity and 0.0368 for Dense over 112,590 items. Dense was better here but still missed most next items. The users averaged 2.293 train interactions; that is a concrete sparsity clue, not proof of a specific failure cause. The fixed BGE encoder was not trained by this project, and no test-driven tuning was performed. Further analysis should segment by history length and product popularity using validation first.

## What is actually implemented?

The local data normalization, split, NumPy exact index, history pooling, popularity counter and evaluation protocol are project code. Sentence Transformers and optional FAISS are library calls. BGE is an external pretrained encoder. RQ-VAE, Transformer generation and tool-policy SFT are not yet trained or evaluated.
