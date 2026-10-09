# M2 original RQ-VAE plan and observed outcome

This section preserves the original plan made before the laboratory run. Later on 2026-10-01, the full catalog was trained and independently verified, but the learned codebooks collapsed. The separate residual K-Means baseline and both full results are recorded in [results_m2.md](results_m2.md). Do not present the plan below as an unexecuted proposal or the RQ-VAE as a successful Semantic ID tokenizer.

## Proposed full run

- Input: the existing 112,590 × 384 L2-normalized BGE product vectors in `data/processed/All_Beauty/index/item_embeddings.npy` (about 173 MB). The associated item IDs and index metadata pin dataset revision `2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e` and BGE revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`. No new dataset or model download is required.
- Model: project-implemented MLP encoder 384→256→64, three learned residual codebooks with 256 entries of dimension 64 each, and MLP decoder 64→256→384. Total **279,488 parameters**. Quantization uses nearest-code assignment, straight-through reconstruction gradients, codebook loss and commitment loss (`0.25`). Codebooks are initialized from **training-item** latents using k-means++ seeding; no validation or test target is used for initialization.
- Item-level deterministic reconstruction split: seed 42, 90% train (about 101,331 items), 10% validation (about 11,259 items). This is separate from the M1 user-sequence train/validation/test split. The static metadata catalog visibility assumption applies. Validation reconstruction MSE selects the best checkpoint; user next-item test interactions do not influence this training.
- Optimizer: Adam, learning rate `0.001`, batch size `512`, maximum `20` epochs, about `198` training steps per epoch or `3,960` total. Device: local RTX 4060 Laptop GPU with 8 GiB VRAM; CPU fallback is available. A rough runtime estimate is **10–60 minutes**, depending on actual throughput and GPU contention. This is an estimate, not a measured full-run duration. Expected checkpoint size is small (<10 MB); item-to-token JSON and logs add tens of MB. Existing vectors are read from the project directory.
- Executed command: `python scripts/train_rqvae.py --device cuda --output-dir outputs/m2_rqvae_full`. The run recorded source checksums, code revision, package versions, seed, split item IDs, losses per epoch, best checkpoint and item Semantic ID map.

## Acceptance checks

1. Check finite train/validation reconstruction losses and save/load equality for the selected checkpoint.
2. Report per-level used-code count, utilization and perplexity on the entire item catalog, plus unique code count, collision groups, colliding items and largest group. Do not rename a collapsed codebook as a successful Semantic ID result.
3. Every saved code must map to at least one real catalog item. A collided code maps to all matching IDs; later M3 ranking must resolve ambiguity without looking up the held-out target.
4. Preserve the complete training log and report failure or degraded results. RQ-VAE reconstruction/usage is not itself a next-item recommendation improvement; M3 will evaluate that separately.

## CPU smoke evidence

Both smoke runs used the same 128 real item vectors, two epochs, two levels of 16 codes, and a 64→16 hidden/latent configuration on CPU. Random codebook initialization used **1/16 codes at each level**, with all 128 items colliding. Training-item latent initialization used **9/16 and 10/16 codes**, with 26 unique code sequences and 119 items in collision groups. Best validation reconstruction MSE in the latter run was `0.003603`. These are pipeline diagnostics, not a full-catalog experiment or a performance claim.

## Full-run outcome

The full RQ-VAE used 3/84/20 codes per level, yielded 1,718 unique three-token IDs for 112,590 items and put 112,405 items in collision groups. Its checkpoint, source hashes, split and reconstruction MSE were independently verified. Low validation MSE alone did not satisfy the utilization/collision acceptance checks. Preserve this negative result as a control when evaluating the separately named PCA-whitened residual K-Means baseline.

## Stabilized learned RQ-VAE implementation — 2026-10-09

The new explicit `pca_ema` variant is a trained neural encoder/decoder plus residual vector quantizer, **not the separate fixed PCA/RQ-KMeans baseline**. Training-item PCA initializes trainable linear skip projections; zero-initialized residual MLPs, spherical latents, cosine reconstruction, and variance/covariance regularization stabilize the representation. Three EMA codebooks learn from training residuals, with unused codes restarted from the current training batch. Autoencoder warmup precedes train-only k-means++/Lloyd codebook initialization. Encoder/decoder gradients and real updates, EMA updates and dead-code replacement are tested. Validation and public encoding cannot update EMA state. The default remains `legacy`; old six-field checkpoints still load without changing their saved state layout.

This is an independently implemented stabilization bundle, not an original algorithm or a one-intervention causal result. References: [VQ-VAE](https://arxiv.org/abs/1711.00937), [RQ-VAE/RQ-Transformer](https://arxiv.org/abs/2203.01941) and the official [EMA/residual-quantization implementation](https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/models/rqvae/quantizations.py), plus [VICReg variance/covariance regularization](https://arxiv.org/abs/2105.04906). No upstream source file was copied; the variance/covariance terms are an adaptation, not a full paired-view VICReg implementation. Their published results do not prove effectiveness on this product catalog.

Code/config: `bf647aa`, `configs/semantic_id_stable.yaml`. Local targeted M2 tests: 14 passed. Complete isolated lab95 suite: **181 passed in 7.25 seconds**. No dependency changes or full-catalog training were performed. The malformed personal SSH configuration was bypassed for these connections, not edited. The isolated checkout is `/mnt/wangyuqing/genrec_upload/m2_stable_bf647aa/GenRecAgent` and uses the existing dedicated project interpreter.

| Actual CPU smoke | Catalog / validation size | Code shape | Catalog used codes | Catalog perplexity | Unique codes | Val MSE | Gate result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Small variant | 1,024 / 102 | 3×32, latent 32 | 32 / 32 / 32 | 30.17 / 26.75 / 26.27 | 960 | 0.00061179 | validation and catalog pass |
| Full-shape variant | 4,096 / 410 | 3×256, latent 64 | 256 / 256 / 255 | 224.73 / 132.41 / 62.84 | 4,018 | 0.00054448 | validation usage/perplexity and catalog third-level perplexity fail |

The full-shape CPU run executed one warmup and three quantized epochs; quality-aware selection retained epoch 1. Its saved checkpoint has eight EMA-update steps, not a claim that only one epoch executed. The full log preserves all three epochs. The 1,024 run used two warmup and three quantized epochs. Timings were 2.91 and 3.56 seconds respectively. Sample scopes/latent dimensions differ; neither is a full-catalog repair result or a fair numerical replacement for the old 112,590-item model. The third-level catalog perplexity 62.84 is below the **predeclared 64** cutoff and remains failed; it was not rounded upward or accepted by lowering the threshold.

Both returned checkpoints were independently reloaded and re-encoded on every sampled item: **0 mapping mismatches over 1,024 and 4,096 items**, decoded MSE matching saved metrics within 1e-7, and no EMA state changes. Artifacts: `outputs/m2_stable_lab95_checks`; returned archive SHA256 `c9b5651c1c7001d8592d3b638817c280ff9df7a6ece5096d306ee24435ce47a9`. The failed initial bundle checkout log is preserved separately; the successful suite checked out `main` explicitly.

### Proposed full run — approval still required

Reuse the original 112,590×384 item matrix (embedding SHA256 `631ac5004acce6d9ab4f1ef9c4b3e350390d3f10ba9ef5a86fcc9bce819a2b94`), seed 42 and exactly the same 101,331/11,259 item split. Initialize a **fresh** model from PCA fitted on at most 32,768 training items, not from either smoke checkpoint. Train 198,016 neural parameters plus 49,152 EMA-updated codebook elements. Use three levels of 256, latent 64, hidden 256, batch 512, LR 0.0003, three warmup epochs and at most twenty quantized epochs (approximately 4,554 optimizer updates total). No user next-item test labels or test recommendation results enter fit/selection. No new dataset/model download, GPU package change or paid API is needed. Proposed device: lab95 RTX 3090 with at least 1 GiB free; never terminate other users' work. Estimated **1–10 minutes** including initialization and full encoding is a rough estimate, not an observed full stabilized runtime; the old GPU run took 27.65 seconds, but the new objectives and updates differ.

The frozen gates require each level to use at least 50% of its codebook, perplexity at least 25% of codebook size (64 at K=256), unique-code ratio at least 50%, largest group fraction at most 2%, and reconstruction MSE at most 0.000703 (about 1.25× original validation MSE). Validation-admissible checkpoints are selected by validation MSE. If none passes, a diagnostic fallback is saved with an explicit failed flag; it is not a successful tokenizer. Full-catalog and selected-validation gates must **both** pass before adoption. These thresholds establish a minimum noncollapsed baseline, not superiority over the existing K-Means tokenizer or a recommendation gain.

After explicit confirmation, command: `python scripts/train_rqvae.py --config configs/semantic_id_stable.yaml --output-dir outputs/m2_rqvae_stable_full`. The script protects fresh project-output paths and preserves all old RQ-VAE/K-Means/SFT/M3 artifacts. A new token map must **not** be combined with the existing Transformer checkpoint; M3 prefixes and model need separately rebuilt/retrained once M2 is actually accepted.
