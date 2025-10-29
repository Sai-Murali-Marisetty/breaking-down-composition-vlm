#!/bin/bash
#SBATCH --job-name=llava_eval
#SBATCH --output=logs/llava_eval_%j.out
#SBATCH --error=logs/llava_eval_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=a100-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=smarisetty@stonybrook.edu

set -e
set -x

echo "Job started: $(date)"
echo "Node: $(hostname)"

module load cuda121

cd /gpfs/scratch/smarisetty/breaking-down-composition-vlm
source venv/bin/activate

# Set cache to scratch
export HF_HOME=/gpfs/scratch/smarisetty/.cache/huggingface
export TRANSFORMERS_CACHE=/gpfs/scratch/smarisetty/.cache/huggingface
mkdir -p $HF_HOME

nvidia-smi
python --version

python evaluate_llava_baseline.py <<EOF
2
EOF

echo "Completed: $(date)"