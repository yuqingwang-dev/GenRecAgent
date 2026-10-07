#!/usr/bin/env bash
# The caller must have obtained the user's approval for the bounded real training run.
set -euo pipefail
PYTHON="${1:?Pass the existing GenRecAgent environment Python executable}"
MODE="${2:-}"
[[ "$MODE" == "--approved" || "$MODE" == "--check-only" ]] || { printf 'Use --check-only for read-only checks; real training needs explicit approval.\n' >&2; exit 64; }
cd "$(dirname "$(realpath "$0")")/.."
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PREFLIGHT=$("$PYTHON" -u scripts/check_recovery_run.py)
printf '%s\n' "$PREFLIGHT"
[[ "$MODE" != "--check-only" ]] || exit 0
mkdir outputs/recovery_job
trap 'printf "%s\n" "$?" > outputs/recovery_job/exit_code.txt' EXIT
printf '%s\n' "$PREFLIGHT" > outputs/recovery_job/preflight.json
"$PYTHON" -u - <<'PY'
import json, torch
free, total = torch.cuda.mem_get_info()
if not torch.cuda.is_bf16_supported() or free < 5.5 * 1024**3:
    raise RuntimeError("BF16 and 5.5 GiB free required; never interrupt other GPU users")
print(json.dumps({"gpu":torch.cuda.get_device_name(0),"free_bytes":free,"total_bytes":total}), flush=True)
PY
"$PYTHON" -u scripts/train_tool_sft.py --config configs/tool_sft_recovery_v3.yaml \
    --output-dir outputs/tool_sft_recovery_v3 > outputs/recovery_job/train.log 2>&1
"$PYTHON" -u scripts/evaluate_tool_agent.py --policy compact_sft \
    --adapter outputs/tool_sft_recovery_v3/adapter --split validation --max-tasks 72 \
    --output-dir outputs/recovery_v3_sft_validation > outputs/recovery_job/validation.log 2>&1
# Training/checkpoint selection is over before the new held-out test is opened.
for policy in fixed compact_prompt compact_sft; do
    extra=()
    if [[ "$policy" == compact_sft ]]; then
        extra=(--adapter outputs/tool_sft_recovery_v3/adapter)
    fi
    "$PYTHON" -u scripts/evaluate_tool_agent.py --policy "$policy" "${extra[@]}" \
        --data-dir data/processed/tool_sft_recovery_v3 --split test --max-tasks 72 \
        --output-dir "outputs/recovery_v3_test/$policy" \
        > "outputs/recovery_job/test_$policy.log" 2>&1
done
printf '0\n' > outputs/recovery_job/exit_code.txt
tar -czf outputs/recovery_v3_results.tar.gz -C outputs \
    tool_sft_recovery_v3 recovery_v3_sft_validation recovery_v3_test recovery_job
sha256sum outputs/recovery_v3_results.tar.gz
