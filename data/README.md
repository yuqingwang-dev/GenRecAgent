# Data directory

`raw/` and `processed/` are reserved for future datasets and generated artifacts. They are ignored by Git so that credentials, large files, and experiment outputs are not committed accidentally.

M1 uses two explicit official All_Beauty Parquet shards. `scripts/prepare_amazon.py --download` stores Hugging Face cache only in `data/raw/hf_cache`; processed JSONL and a provenance manifest go under `data/processed/All_Beauty`. No dataset is shipped with the repository. Local fixture tests are labeled `synthetic_fixture` in their manifest.
