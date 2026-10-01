# Resume evidence ledger

Only work with a reproducible artifact and an actual result should become a resume claim.

| Planned claim | Code / command | Current evidence | Status |
| --- | --- | --- | --- |
| Chronological Amazon Reviews preparation | `data/amazon.py`, `scripts/prepare_amazon.py` | official All_Beauty revision and hashes in `data/processed/All_Beauty_full_optimized/manifest.json`; 8,151 eligible users documented in `results_m1_full.md` | verified on full eligible-user set |
| Dense and popularity next-item baselines | `retrieval/dense.py`, `retrieval/popularity.py`, `scripts/evaluate_dense.py` | real validation/test metrics in `outputs/m1_all_beauty_full/metrics.json` and summary in `results_m1_full.md`; clean code commit `42e1b19` | verified on full eligible-user set |
| RQ-VAE semantic tokens | `semantic_id/{rqvae,quantizer,training,codebook}.py`, `scripts/train_rqvae.py` | full-catalog checkpoint and logs in `outputs/m2_rqvae_full/`, independently reloaded; 1,718 unique IDs and severe collapse in `results_m2.md` | full training verified; quality failed |
| PCA-whitened residual K-Means baseline | `semantic_id/{projection,rqkmeans}.py`, `scripts/train_rqkmeans.py` | full-catalog codebooks and mapping in `outputs/m2_rqkmeans_full/`; 97,231 unique codes in `results_m2.md`; clean code commit `09e9c56` | full tokenizer baseline verified; recommendation quality unmeasured |
| M3 generative retrieval | `data/sequences.py`, `retrieval/{semantic_trie,generative,transition,transformer}.py`, `scripts/{prepare_m3_sequences,evaluate_transition,train_semantic_transformer}.py` | train-only code-transition full-catalog test Recall@20 0.007852 in `results_m3_transition.md`; Transformer 128/32 one-epoch CPU smoke only | transition baseline verified; Transformer full training and retrieval evaluation pending |
| Qwen tool decision and SFT | M4/M5 traces and adapter | none | planned |
| Constraint-aware agent and ablations | M4/M6 task set and metrics | Phase 0 mock only | planned |

The M1 data pipeline and exact retrieval code are project implementations. BGE, Sentence Transformers, FAISS, NumPy and the Amazon dataset are external components; using them is not an original algorithm claim. Add model/data licenses and version evidence before any public release of derived artifacts.
