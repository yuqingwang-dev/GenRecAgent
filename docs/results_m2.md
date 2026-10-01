# M2 Semantic ID comparison: trained RQ-VAE and RQ-KMeans baseline

Date: 2026-10-01. Both runs used the same 112,590 L2-normalized BGE item vectors, identical 101,331/11,259 item-level training/validation IDs (seed 42), and a 3-level, 256-code-per-level discrete ID. The input embedding SHA-256 is `631ac5004acce6d9ab4f1ef9c4b3e350390d3f10ba9ef5a86fcc9bce819a2b94`. The item split lists were compared directly and are equal. No user next-item labels or test hits were used to fit or select either tokenizer.

| Metric | RQ-VAE (original run) | PCA-whitened residual MiniBatch K-Means (separate baseline) |
| --- | ---: | ---: |
| Validation MSE in original 384-D space | 0.00056240 | 0.00045926 |
| Used codes, levels 0/1/2 | 3 / 84 / 20 of 256 | 256 / 256 / 256 of 256 |
| Effective perplexity, levels 0/1/2 | 2.89 / 70.94 / 5.88 | 225.71 / 225.56 / 231.41 |
| Unique 3-token sequences | 1,718 | 97,231 |
| Items in collision groups | 112,405 (99.84%) | 24,794 (22.02%) |
| Largest collision group | 806 | 67 |
| Items in groups larger than 20 | 106,615 | 479 |

The RQ-VAE run used the project MLP encoder/decoder and trainable residual codebooks on the laboratory GPU for 20 epochs. Its checkpoint and code mapping were independently loaded and checked; the low code usage is a real trained outcome, not a file-transfer error. The K-Means baseline used a 64-D PCA whitening projection **fitted on training items only**, followed by three residual MiniBatch K-Means codebooks. Its 384-D validation MSE was measured after inverse projection. PCA and the codebooks were saved as safe `.npz` files; a local 1,000-item re-encode check found zero token mismatches, and the full validation MSE matched the saved metric exactly.

The two methods differ in more than one design choice (learned MLP versus PCA, SGD quantization versus MiniBatch K-Means). This is an M2 tokenizer comparison, **not** evidence that a single intervention fixed RQ-VAE or improved next-item recommendation. The original RQ-VAE failure remains a separate, reproducible baseline. A raw-vector residual K-Means 4,096-item smoke run also showed heavy imbalance (278 unique codes with 3×64 codebooks); PCA whitening plus 3×64 improved to 2,025 unique codes, and the planned 3×256 configuration yielded 4,044/4,096 unique codes on the same subset. These smoke configurations are not directly comparable to the full 3×256 result.

Remaining K-Means collisions are explicitly preserved: `SemanticCodebook.lookup(code)` returns all catalog item IDs sharing that code. M3 must expand, deduplicate and rank these candidates using permitted history or train-only statistics; it must never use the held-out target to select an item within a group. A semantic code by itself is not a unique item identifier for the 24,794 colliding products. No generative retrieval Recall/NDCG has been measured yet.

Reproducibility: original RQ-VAE artifact `outputs/m2_rqvae_full/` records code commit `54af100`, 20 epochs, PyTorch `2.7.1+cu118` and its source hashes. K-Means artifact `outputs/m2_rqkmeans_full/` records clean code commit `09e9c56`, NumPy `2.4.6`, scikit-learn `1.9.1`, per-level fit logs, PCA/codebook arrays, source hashes, metrics and item splits. Both output directories are Git-ignored and were preserved unchanged.
