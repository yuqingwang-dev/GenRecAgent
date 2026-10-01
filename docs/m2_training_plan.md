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
