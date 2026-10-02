# M3 small Transformer: pre-run plan (historical)

This document records the gate and risks identified before full training. The full 2026-10-02 run and its negative retrieval result are reported in [results_m3_transformer.md](results_m3_transformer.md); the pending-status statements below describe the earlier planning state, not the current project status.

At planning time, model, target-only label masking, causal attention, batched code scorer, checkpointing and a shared full-catalog evaluator were implemented. **Only a 128-train/32-validation, one-epoch CPU smoke run had executed**; its validation token accuracy was zero and its 10-user retrieval smoke had zero hits with a four-code budget. Neither was a formal model result.

## Proposed full training

- Frozen inputs: `data/processed/All_Beauty_m3/{train,validation}.jsonl` and `outputs/m2_rqkmeans_full/item_semantic_ids.json`, whose hashes are checked against the M3 manifest. Only **10,969 train-prefix examples** update weights. All **8,151 validation examples** select the checkpoint by target-token negative log-likelihood. The user next-item test file is not read by training.
- Default model in `configs/generative.yaml`: decoder-only causal Transformer, 2 layers, width 128, 4 heads, feedforward width 256, dropout 0.1, full vocabulary of 768 learned-code tokens plus 3 special markers. **473,731 parameters** with a maximum sequence length of 83 tokens (20 history items × four markers/tokens plus prediction prefix). Labels supervise only the next item's three code tokens; history and padding positions use `-100`.
- Default optimizer/budget: AdamW, learning rate 0.0005, batch 128, 20 epochs, about **86 train steps per epoch / 1,720 total**. The model has no pretrained weight download. Target hardware can be the local RTX 4060 8 GiB or the laboratory GPU after its environment is verified. Full runtime has not been measured; allow minutes to tens of minutes and time the first epoch before promising a duration. Model weights are small; batching and subsequent 8,151-user constrained decoding may dominate runtime.
- Proposed command after training approval: `python scripts/train_semantic_transformer.py --device cuda --output-dir outputs/m3_transformer_full`. The output directory must be empty. The run stores `transformer.pt`, vocabulary, epoch log, best-validation checkpoint metrics, source hashes, Git commit and device/version information.
- After selecting checkpoint/beam on validation, use `scripts/evaluate_semantic_transformer.py` on validation and then the held-out test set with a fresh output directory. Full evaluation refuses a checkpoint marked as a smoke/subset run. It shares the train-only collision ordering and Recall/NDCG evaluator with the code-transition baseline.

## Risks and acceptance checks

The 10,969 prefixes come from only 2,909 users with at least two train items; 5,242 eligible users have a single train item and contribute no training prefix. Only 31.33% of test targets' *full codes* occur as train-prefix target codes. Constrained decoding can output other valid catalog codes, but generalization cannot be assumed. Record validation learning curves, valid/duplicate/insufficient candidates, collisions, Recall/NDCG@5/10/20 and latency. Keep a weak or regressed Transformer result as evidence; do not tune on test or confuse the CPU smoke with a full run.
