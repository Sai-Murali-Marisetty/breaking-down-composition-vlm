#!/bin/bash
#SBATCH --job-name=qwen2vl
#SBATCH --output=logs/qwen2vl_%j.out
#SBATCH --error=logs/qwen2vl_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=a100-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

set -euo pipefail

echo "Testing Qwen2-VL on Winoground"
echo "Started: $(date)"

# If PyTorch was installed via pip (recommended), do NOT load a CUDA module that can conflict.
# module load cuda121   # <- leave commented unless you built PyTorch from source against this module

cd /gpfs/scratch/smarisetty/breaking-down-composition-vlm
source venv/bin/activate

export HF_HOME=/gpfs/scratch/smarisetty/.cache/huggingface
export TRANSFORMERS_CACHE=/gpfs/scratch/smarisetty/.cache/huggingface
mkdir -p "$HF_HOME" logs results

# Debugging aids (uncomment if you want precise CUDA error lines)
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

echo "nvidia-smi:"
nvidia-smi || true

# Ensure qwen_vl_utils present (idempotent)
pip install --upgrade --no-cache-dir qwen-vl-utils

# Optional: clear any corrupted model cache for this model
# rm -rf "$HF_HOME/hub/models--Qwen--Qwen2-VL-7B-Instruct" || true

python evaluate_qwen2vl.py

echo "Completed: $(date)"
