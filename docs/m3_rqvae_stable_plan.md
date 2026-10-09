# M3 migration to stabilized RQ-VAE codes: preparation and approval gate

Date: 2026-10-09. **The separately approved full Transformer training/evaluation, independent audits and local checksum return are complete.** The user explicitly confirmed “批准 M3 新编码训练及评测”; [actual results](results_m3_rqvae_stable.md) record one 20-epoch run (38.01 seconds) and both full retrieval splits (273.682 seconds). The historical proposal below was frozen before launch. The existing K-Means M3 samples, checkpoint, saved metrics, router and delivery defaults are unchanged; small/mixed new relevance changes do not justify promotion.

## Executed preparation

At clean source commit `915364493ad27c76d8360654b1b7e7ef1fb1671e`, `configs/generative_rqvae_stable.yaml` prepared an independent dataset under `data/processed/All_Beauty_m3_rqvae_stable/`. Input is the [accepted stabilized neural M2 map](results_m2.md), not the collapsed original or the separate K-Means tokenizer. Source histories and map are pinned in the new config; the new mapping's catalog quality gate must pass. Existing `configs/generative.yaml` and its defaults are not modified.

| Split | Examples | Chronological rule |
| --- | ---: | --- |
| Train | 10,969 | Train-period prefixes predict a later train-period event only |
| Validation | 8,151 | Observed train history predicts validation event |
| Test | 8,151 | Observed train + validation history predicts test event |

Histories retain the most recent 20 items. There are 8,151 users, of whom 2,909 contribute training prefixes and 5,242 contribute none but remain in both evaluation denominators. Per-user leave-last-out/static-catalog limitations remain; this is not a global-time split or an online click/purchase benchmark.

Preparation command: `python scripts/prepare_m3_sequences.py --config configs/generative_rqvae_stable.yaml`. Successful fresh preparation exited 0, with a recorded clean Git state and complete sample hashes. A first attempt stopped during `train.jsonl` writing without a Python traceback; its incomplete output is retained separately at `data/processed/All_Beauty_m3_rqvae_stable_first_attempt_20261009/`, never used for training. It was not silently overwritten or attributed to an unproven cause.

Independent audit command: `python scripts/audit_m3_sequences.py --config configs/generative_rqvae_stable.yaml --reference-sequences data/processed/All_Beauty_m3 --output-dir outputs/m3_rqvae_stable_preparation`. It **does not call the sample builder**: all 27,271 rows match raw-history positions, chronological windows and new item/code assignments; every row's user/split/history/target item also matches the old control. It validates all sample hashes, all 112,590 catalog assignments and collision statistics. Validation-only train-target-code coverage is 2,856/8,151 (diagnostic, not a model score); no test-code-coverage diagnostic or test-guided filtering/tuning was performed. Test rows were read only by this offline integrity audit, not by training or model selection.

Audit evidence: `outputs/m3_rqvae_stable_preparation/audit.json`; sample provenance: `data/processed/All_Beauty_m3_rqvae_stable/manifest.json`. Full local regression: **190 passed in 12.93 seconds**, exit 0; Ruff passed. Tests cover corrupted future contexts/codes/splits/counts even if the file hash is updated, source fingerprint/quality mismatches, output boundary/no-overwrite checks, and refusal of unapproved training before loading inputs or Torch.

## Frozen fingerprints

- New map: `f16acf7acaefa72a64b06df6034ae055beb0c971c6e19076b82fc6eb707906e5`.
- Original histories: `b8d1bde5a643fa64b8791ac0cf368eb691bb8eba0bba1a80d03e4acaf128eb6b`.
- New train: `00bc5b8d869df86e7f16a3b3010ba5790c1466367329d752550ebcc9c95bf0c4`.
- New validation: `43a0d9f0fd46b77a134743983826dd297c04a8fd4c85b380e98fd94641ce1c90`.
- New test: `8e885c7e822b09af386030ceae6ed009d7fda8a24868e3654e0860fc8ea1c7c0`.
- Frozen new config: `9cacfa4d62c2e3d7fa7c2cdc65d5796e18260ab1793271eeb0eae6ed6a0c95a7`.

Old M3 checkpoint remains `dba260171051311a4e214947bfdaec2449e8cabad0ea6e35907076fb08422dd3`; old K-Means map remains `f394e4ce1a39ec7eb078a9dc8141c3438359c52e7fc6a8c40be1e42e05c45c9c`. Old train/validation SHA256 remain `43fcf7c637fa13b2a6a77b16673eb7552275619e94945638f56aa94f3fa6d242` / `c599709d510d015187a3ee254be7b7898341d8478e0380148353911b2d0c2f0d`. These were checked after preparing/auditing the new data. Existing delivery must not substitute the new code map into its old checkpoint.

## Frozen one-run proposal — approved and executed 2026-10-09

Keep the original Transformer architecture, optimization and decoder settings fixed to make the tokenizer change interpretable. This is one seed/one fresh run, not a search:

- **473,731 parameters**; 2 causal layers, width 128, 4 heads, feedforward 256, dropout 0.1, vocabulary 768 code tokens + 3 markers, max sequence 83. Initialize a fresh model; do not resume the old checkpoint. Only the target's three code tokens are supervised; history/padding use `-100`.
- AdamW LR 0.0005, batch 128, gradient clip 1.0, seed 42, **20 epochs**, 86 train steps/epoch (**1,720** total). Only the 10,969 train prefixes update weights. All 8,151 validation examples select the minimum target-NLL checkpoint. The training entry reads neither `test.jsonl` nor raw held-out histories.
- Target: **lab95 RTX 3090**, reusing the dedicated existing interpreter. Check CUDA and at least **2 GiB free VRAM** before launch; this is a conservative reservation, not a measured peak. Never terminate another user's job. No pretrained model, dataset download, new package installation or paid API is proposed. Prior equal-shape training took 115.66 seconds; estimate **2–5 minutes for training**, **5–20 minutes for both full retrieval splits**, depending on load. These are estimates, not new measured runtimes.
- New server/local training output: `outputs/m3_transformer_rqvae_stable_full/`; retrieval output: `outputs/m3_transformer_rqvae_stable_evaluation/`. Use a new isolated server checkout and fresh paths, reuse existing inputs only after fingerprint checks, and return checksum-verified weights/logs/evaluation/audit files locally. Old model/data/results stay preserved.

The approved command was `python scripts/train_semantic_transformer.py --config configs/generative_rqvae_stable.yaml --approved`; it has already completed and must not be relaunched into existing outputs. The new config's `approval_required: true` refuses an unapproved launch before input/model loading. The CLI flag records acknowledged approval; it is not permission to bypass confirmation or authorize further model searches. No new approval is needed to accept this already completed run's results.

## Frozen evaluation and handoff

After training, independently verify checkpoint load, finite tensors/losses, full logs/selection and map/sample fingerprints. Then run `python scripts/evaluate_semantic_transformer.py --config configs/generative_rqvae_stable.yaml --split both --device cuda` once with the fixed selected checkpoint. Beam size 20, max codes 20, K=5/10/20; expand collisions using train-only popularity, exclude observed history, deduplicate, and keep one target per user in the full 8,151 denominator. Invalid/duplicate codes/items, shortfalls, collisions, Recall/NDCG and latency must be reported. No dense fusion or test-directed beam/epoch/settings search is included in this approval proposal.

Compare against the preserved [old M3 Transformer](results_m3_transformer.md) and [M1 Popularity/Dense](results_m1_full.md) under the same catalog/user protocol. Existing baseline relevance metrics are frozen references; historical cross-hardware latency is not a like-for-like speed comparison. The existing test set has already been observed for earlier baselines, so it is **not a newly independent holdout**; retain fixed settings and report that limitation. Better M2 reconstruction/usage does not guarantee higher Recall/NDCG. Keep regressions and do not switch production/demo defaults based on tokenizer acceptance alone.
