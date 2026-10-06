#!/usr/bin/env bash
# Run the approved one-epoch experiment without depending on an open SSH terminal.
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/.."
PYTHON="${1:?Pass the GenRecAgent environment Python executable}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
mkdir -p outputs/lab95_job
trap 'printf "%s\n" "$?" > outputs/lab95_job/exit_code.txt' EXIT
"$PYTHON" -u - <<'PY'
import hashlib, json, pathlib, torch
root = pathlib.Path.cwd()
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
free, total = torch.cuda.mem_get_info()
print(json.dumps({"gpu": torch.cuda.get_device_name(0), "free_bytes": free,
                  "total_bytes": total, "torch": torch.__version__}), flush=True)
if free < 5.5 * 1024**3:
    raise RuntimeError("Less than 5.5 GiB free; do not disrupt other GPU users")
manifest = json.loads((root / "outputs/m4_qwen_download/manifest.json").read_text())
snapshot = root / "artifacts/model_cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots" / manifest["revision"]
for name, record in manifest["files"].items():
    with (snapshot / name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == record["sha256"], name
print("Pinned base snapshot hashes verified", flush=True)
PY
"$PYTHON" -u scripts/train_tool_sft.py --output-dir outputs/tool_sft_lora \
    > outputs/lab95_job/train.log 2>&1
for policy in fixed prompt sft; do
    extra=()
    if [[ "$policy" == sft ]]; then
        extra=(--adapter outputs/tool_sft_lora/adapter)
    fi
    "$PYTHON" -u scripts/evaluate_tool_agent.py --policy "$policy" "${extra[@]}" \
        --output-dir "outputs/lab95_comparison/$policy" \
        > "outputs/lab95_job/$policy.log" 2>&1
done
for ablation in no_recovery no_finish_verification; do
    "$PYTHON" -u scripts/evaluate_tool_agent.py --policy fixed --ablation "$ablation" \
        --output-dir "outputs/lab95_comparison/$ablation" \
        > "outputs/lab95_job/$ablation.log" 2>&1
done
printf "0\n" > outputs/lab95_job/exit_code.txt
tar -czf outputs/lab95_sft_results.tar.gz -C outputs \
    tool_sft_lora lab95_comparison lab95_job
sha256sum outputs/lab95_sft_results.tar.gz
