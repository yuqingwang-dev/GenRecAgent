# M2 Semantic ID comparison: trained RQ-VAE and RQ-KMeans baseline

Original comparison date: 2026-10-01; these historical results are preserved below. The approved stabilized neural full result is appended in the 2026-10-09 section. Both original runs used the same 112,590 L2-normalized BGE item vectors, identical 101,331/11,259 item-level training/validation IDs (seed 42), and a 3-level, 256-code-per-level discrete ID. The input embedding SHA-256 is `631ac5004acce6d9ab4f1ef9c4b3e350390d3f10ba9ef5a86fcc9bce819a2b94`. The item split lists were compared directly and are equal. No user next-item labels or test hits were used to fit or select either tokenizer.

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

## Stabilized learned RQ-VAE — 2026-10-09

The user explicitly approved one full run. A **fresh** `pca_ema` RQ-VAE trained on the entire original catalog and identical item split at clean source commit `bf647aa1e7733f70c8359c463c9c4ced6addd700`. PCA uses 32,768 training items only to warmstart **trainable** neural skip projections; residual MLPs learn jointly with EMA-updated residual codebooks after autoencoder warmup. This is not a relabeled fixed PCA/RQ-KMeans baseline. Multiple design choices changed together; this is a stabilization bundle comparison, not a one-factor causal attribution or a novel algorithm claim. Architecture, references, preset gates and the earlier failed full-shape smoke remain in [m2_training_plan.md](m2_training_plan.md).

| Full-catalog metric | Original neural RQ-VAE | Separate RQ-KMeans | Stabilized learned RQ-VAE |
| --- | ---: | ---: | ---: |
| Validation reconstruction MSE, 384-D | 0.00056240 | 0.00045926 | 0.00041323 |
| Catalog used codes, levels 0/1/2 | 3 / 84 / 20 | 256 / 256 / 256 | 256 / 256 / 256 |
| Catalog perplexity, levels 0/1/2 | 2.89 / 70.94 / 5.88 | 225.71 / 225.56 / 231.41 | 237.14 / 236.76 / 246.07 |
| Unique three-token codes | 1,718 | 97,231 | 93,350 |
| Items in collision groups | 112,405 (99.84%) | 24,794 (22.02%) | 30,055 (26.69%) |
| Largest collision group | 806 | 67 | 67 |

Selected epoch **20**, after all **3 warmup + 20 quantized epochs** executed. Train/validation item counts are 101,331/11,259; batch 512, Adam LR 0.0003, seed 42, latent 64, hidden 256, 3×256 codes. Total model parameters: 247,168 (198,016 neural gradient-trained parameters + 49,152 EMA codebook elements). Selected checkpoint EMA updates: 3,960, with dead-code resets 17/213/291. Actual lab95 RTX 3090 / Torch 2.7.1+cu118 / NumPy 2.2.6 runtime: **54.789 seconds** in the pre-save record, **55.40 seconds** including saving/logging, exit **0**. No additional run, dependency installation, dataset/model download or paid API was required.

The **predeclared gates were unchanged**: each level usage ≥128 and perplexity ≥64, unique ratio ≥50%, largest group fraction ≤2%, reconstruction MSE ≤0.000703. Selected validation uses 256/254/254 codes, perplexity 233.6739/233.1335/242.1741, unique ratio 96.7404%, largest group 0.07105%, MSE **0.00041322541506614815**. Catalog unique ratio is 82.9114%, largest group 0.05951%, MSE **0.00040518768032502293**. **Validation and catalog both pass.** Selection is minimum validation MSE among admissible epochs, not catalog/test recommendation tuning. The old 4,096-item smoke's third-level perplexity failure is retained unchanged.

Independent audit reloaded `rqvae.pt` and forward/encode/decode-checked all **112,590** catalog vectors on CUDA: **zero saved-map mismatches**, independently accumulated float64 catalog MSE 0.000405187669239414 and validation MSE 0.00041322540336783753 (agreement within 1e-7). Quantizer/EMA buffers did not change during evaluation or train-mode public `encode`. Independent histograms/gates/collisions match; every reverse code lookup preserves the entire collision group. Pinned input/config/source hashes, exact SHA-based split, train-only PCA sample IDs, all finite model tensors/losses, complete epoch logs and checkpoint selection also pass. **33 checks passed**. Final full regression: **181 tests passed in 6.80 seconds**, exit 0.

All artifacts are returned locally under `outputs/m2_rqvae_stable_full_server/`; weights/map/logs/run config/item splits are in its `m2_rqvae_stable_full/` subdirectory. `m2_full_verification.json` and `verify_full_run.py` preserve the independent audit and its executable method. `local_receipt_verification.json` records safe type/path validation of all **16** regular archive entries, every manifest/model checksum, matching local full input, exact train/validation list equality with **both** old baselines and unchanged original M3 checkpoint checksum. Original RQ-VAE/K-Means/SFT/M3 outputs are not overwritten.

- Returned archive SHA256: `964c6c55936a3462473d485cac5dd44108392f4abce627ac837e345bb188557b`.
- Checkpoint SHA256: `5cf96ca752e32633535de5bee2c7abdc254448068aba9c64d32fe51c9984b822`.
- New mapping SHA256: `f16acf7acaefa72a64b06df6034ae055beb0c971c6e19076b82fc6eb707906e5`.
- Frozen config SHA256: `cdd16e4f1c8e9246d23126d23a6b85d0785e80e0b1cbe1a52b9b7cd9227d62bd`.

Reproduction at the pinned training commit, with the existing exact input/interpreter and a **fresh** output directory: `python scripts/train_rqvae.py --config configs/semantic_id_stable.yaml --output-dir outputs/m2_rqvae_stable_full`. Do not rerun into the existing output path. To repeat the returned independent audit, use its original server layout with `PYTHONPATH="$PWD/src"`, CUDA, and a fresh report destination; the preserved script refuses to overwrite its existing report.

**Acceptance boundary:** M2's minimum noncollapsed learned-tokenizer requirement is now met. Reconstruction and code balance improve over the original collapsed neural run, but K-Means still has fewer colliding items and more unique codes. Code usage/reconstruction are not semantic relevance, next-item Recall/NDCG or broad superiority. Current M3, its prepared prefixes and delivery defaults still use the **original K-Means mapping**; its saved Transformer must never be paired with the new code map. Next: build separate train-only prefixes from this new mapping and, after a bounded M3 resource proposal/approval, train a fresh generator and evaluate under the frozen existing catalog/user split. Keep the old M3 result as a control and select settings only on validation.

Subsequent 2026-10-09 update: the separately approved [fresh learned-code M3 Transformer](results_m3_rqvae_stable.md) is now trained, fully evaluated and returned locally with independently regraded candidates. This executes the next step proposed above, without changing the old model/default map. Test Recall@20 adds only two net hits over the old Transformer, while test NDCG/top-5/top-10 decrease; tokenizer acceptance did **not** establish overall recommendation superiority.
