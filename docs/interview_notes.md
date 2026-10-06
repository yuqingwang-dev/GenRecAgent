# Interview notes tied to this repository

## Why these M1 baselines?

Popularity uses only users' train interactions and gives a simple collaborative prior. Dense retrieval uses static product text, mean-pooled vectors of the user's observed items, L2 normalization and exact inner product. The shared catalog and seen-item rules make the two methods comparable. `run_dense_demo.py` is a separate query-to-item task; its output is not evidence for next-item recommendation quality.

## What could make the metrics misleading?

Within-user leave-last-out does not create a global time cut. Static full-catalog metadata may expose descriptions of items unavailable at an earlier event time, and `average_rating` is a scraped aggregate that should not be trained into a historical ranker. Duplicate user-item reviews are collapsed to the earliest event. A missing catalog target stays in the denominator. `source_kind` and input hashes distinguish official data from a synthetic fixture.

## How will a weak result be handled?

Keep the full split and candidate universe fixed. Inspect unreachable targets, sparse train histories, metadata text quality and the difference between frequent and tail items. Any encoder, pooler or fusion choice is selected on validation only; test is held for the final comparison. Do not infer online CTR, conversion or revenue from these offline metrics.

The full eligible-user test result was Recall@20 = 0.006625 for Popularity and 0.036560 for Dense over 112,590 items. Dense was better here but still missed most next items. The 8,151 users averaged 2.346 train interactions; that is a concrete sparsity clue, not proof of a specific failure cause. The fixed BGE encoder was not trained by this project, and no test-driven tuning was performed. The separately retained 5,000-user development subset differs in both user mix and train-only popularity counts. Further analysis should segment by history length and product popularity using validation first.

## What is actually implemented?

The local data normalization, split, NumPy exact index, history pooling, popularity counter and evaluation protocol are project code. Sentence Transformers and optional FAISS are library calls. BGE is an external pretrained encoder. The RQ-VAE implementation is project code and underwent full-catalog training, but its codebooks collapsed. The PCA-whitened residual K-Means baseline uses project orchestration with scikit-learn's PCA and MiniBatchKMeans library implementations; it must not be described as a repaired RQ-VAE or a novel clustering algorithm. The small Transformer and Qwen policy LoRA have now actually trained and been evaluated. Both learned paths remain weak; implementation and controlled measurement are the defensible claims, not a new state-of-the-art recommender.

## What did the first RQ-VAE smoke run show?

Random codebook initialization collapsed both 16-entry levels to one used code, placing all 128 items in one collision group. Seeding codebooks from train-only latents increased usage to 9/16 and 10/16, yet 119 of 128 items still shared codes with another item. The mapping retains every colliding item and does not infer the right one from a held-out target. The smoke configuration has two levels and tiny codebooks; it cannot establish the behavior of the planned three-level full-catalog run.

The full RQ-VAE run confirmed collapse: only 3/84/20 codes were used per level, and 112,405 of 112,590 products collided. Exact duplicate input vectors accounted for only 2,332 products, so the dominant problem was the tokenizer, not identical inputs. A raw-vector residual K-Means smoke test also produced imbalanced later levels. Fitting PCA whitening on training items and then clustering their residuals produced 97,231 unique IDs on the full catalog with the same item split. This is a distinct baseline and trades some pure semantic hierarchy for higher discrete capacity. Remaining collisions require candidate expansion and target-independent ranking in M3. Both the count-transition and subsequent full-trained Transformer use these K-Means codes, not successful RQ-VAE codes.

## Why constrain M3 decoding and retain collisions?

The M2 catalog contains only 97,231 valid three-token paths for 112,590 products. Masking each generation step through a trie avoids fabricated codes. A collided code maps to all corresponding items; the generative-only adapter ranks that group with train-only popularity and filters observed history. Using the held-out target to pick a product would invalidate item-level evaluation. The count scorer and full-trained Transformer both have real results. Only 10,969 train-prefix examples exist and 5,242 users have a single train item, so an oversized sequence model would be difficult to justify. [m3_protocol.md](m3_protocol.md) defines the measured protocol.

The real train-only count-transition scorer reached test Recall@20 0.007852, versus Dense 0.036560. It used a valid-code trie, all 8,151 users, fixed 20-code beam and train-popularity collision ordering; this demonstrates a weak but reproducible sequential baseline. The fully trained 473,731-parameter Transformer reached only 0.002454 test Recall@20. Actual item-level Dense/Transformer fusion was later measured on validation; it did not establish additional Recall@20 hits. Neither a valid code nor low token loss alone proves a better recommendation.

## What did real tool-policy SFT establish?

The model genuinely trained: Qwen2.5-1.5B with 1,089,536 trainable q/v LoRA parameters, one epoch over 608 executed teacher actions on lab95 RTX 3090. Only assistant action labels were supervised. Validation action NLL fell from 0.6410 to 0.0382, but raw policy task contracts improved only from Prompt 0/12 to SFT 4/12; only one of nine feasible tasks actually returned five products. Three passes were correct clarification of infeasible budgets. Fixed orchestration remained 12/12 and is the default demonstration path.

SFT traces still contain 38 unchanged repeated-action rejections and eight required-source precondition failures; some feasible requests are incorrectly clarified. The low supervised token loss does not establish reliable multi-step rollouts. The user/price/goal splits are disjoint, but language templates are reused and state includes normalized constraints. This is a small synthetic action-policy experiment, not broad natural-language personalization or online business uplift. See [the full SFT evidence and limits](results_m5_sft_lab95.md).
