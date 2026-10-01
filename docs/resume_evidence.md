# Resume evidence ledger

Only work with a reproducible artifact and an actual result should become a resume claim.

| Planned claim | Code / command | Current evidence | Status |
| --- | --- | --- | --- |
| Chronological Amazon Reviews preparation | `data/amazon.py`, `scripts/prepare_amazon.py` | official All_Beauty revision and hashes in `data/processed/All_Beauty_full_optimized/manifest.json`; 8,151 eligible users documented in `results_m1_full.md` | verified on full eligible-user set |
| Dense and popularity next-item baselines | `retrieval/dense.py`, `retrieval/popularity.py`, `scripts/evaluate_dense.py` | real validation/test metrics in `outputs/m1_all_beauty_full/metrics.json` and summary in `results_m1_full.md`; clean code commit `42e1b19` | verified on full eligible-user set |
| RQ-VAE semantic tokens | `semantic_id/{rqvae,quantizer,training,codebook}.py`, `scripts/train_rqvae.py` | 128-item CPU smoke and checkpoint tests only; `m2_training_plan.md` records collapse and initialization comparison | pipeline verified; full training pending |
| Transformer generative retrieval and fusion | M3 training/evaluation | none | planned |
| Qwen tool decision and SFT | M4/M5 traces and adapter | none | planned |
| Constraint-aware agent and ablations | M4/M6 task set and metrics | Phase 0 mock only | planned |

The M1 data pipeline and exact retrieval code are project implementations. BGE, Sentence Transformers, FAISS, NumPy and the Amazon dataset are external components; using them is not an original algorithm claim. Add model/data licenses and version evidence before any public release of derived artifacts.
