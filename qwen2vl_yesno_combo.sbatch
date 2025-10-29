#!/bin/bash
#SBATCH --job-name=qwen2vl_yesno_combo
#SBATCH --output=logs/qwen2vl_yesno_combo_%j.out
#SBATCH --error=logs/qwen2vl_yesno_combo_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=a100-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

set -euo pipefail

echo "=== Qwen2-VL YES/NO combo run (Winoground -> ARO) ==="
date

# --- ENV / PATHS ---
PROJECT_ROOT="/gpfs/scratch/smarisetty/breaking-down-composition-vlm"
VENV="$PROJECT_ROOT/venv"

cd "$PROJECT_ROOT"
source "$VENV/bin/activate"

export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TRANSFORMERS_VERBOSITY=error

mkdir -p logs results "$HF_HOME"

echo
echo "nvidia-smi:"
nvidia-smi || true

echo
echo "Python & Torch:"
python - <<'PY'
import torch, sys
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda version (torch):", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("bf16_supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY

# # --- 1) Winoground YES/NO ---
# echo
# echo "=== Running Winoground YES/NO (Qwen2-VL) ==="
# python - <<'PY'
# from evaluate_qwen2vl_yesno_winoground import evaluate_qwen2vl_yesno
# acc = evaluate_qwen2vl_yesno(num_examples=400)
# print(f"[WG-YESNO] Done. Accuracy={acc:.2f}%")
# PY

# --- 2) ARO YES/NO ---
# The evaluator may expect AROLocalLoader; your loader defines AROLoader in load_aro_updated.py.
# This shim attaches AROLocalLoader alias to that module BEFORE importing the evaluator.
echo
echo "=== Running ARO YES/NO (Qwen2-VL, max_examples=2000, balanced) ==="
python - <<'PY'
import importlib, sys
try:
    m = importlib.import_module("load_aro_updated")   # your file with class AROLoader
    if not hasattr(m, "AROLocalLoader") and hasattr(m, "AROLoader"):
        setattr(m, "AROLocalLoader", getattr(m, "AROLoader"))
        print("[ARO] Aliased load_aro_updated.AROLoader -> AROLocalLoader")
except Exception as e:
    print("[ARO] Warning: could not prepare loader alias:", e)

from evaluate_qwen2vl_aro_yesno import evaluate_qwen2vl_aro_yesno
acc = evaluate_qwen2vl_aro_yesno(max_examples=2000, balanced_subset=True)
print(f"[ARO-YESNO] Done. Accuracy={acc:.2f}%")
PY

echo
echo "=== All done ==="
date
