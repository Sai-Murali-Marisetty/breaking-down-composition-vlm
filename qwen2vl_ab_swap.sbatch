#!/bin/bash
#SBATCH --job-name=qwen2vl_ab_swap
#SBATCH --output=logs/qwen2vl_ab_swap_%j.out
#SBATCH --error=logs/qwen2vl_ab_swap_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=a100-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

echo "Qwen2-VL Winoground A/B SWAP"
echo "Started: $(date)"

cd /gpfs/scratch/smarisetty/breaking-down-composition-vlm
source venv/bin/activate

export HF_HOME=/gpfs/scratch/smarisetty/.cache/huggingface
export TRANSFORMERS_CACHE=/gpfs/scratch/smarisetty/.cache/huggingface
mkdir -p "$HF_HOME" logs results

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

nvidia-smi || true
python evaluate_qwen2vl_ab_swap.py

echo "Completed: $(date)"
