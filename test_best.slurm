#!/bin/bash
#SBATCH --job-name=elus-test-best
#SBATCH --partition=gpu
#SBATCH --nodelist=heracles
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_best_%j.out
#SBATCH --error=logs/test_best_%j.err

set -euo pipefail

PYTHON="/home/data/projets-aps/projet6/env_aps/bin/python3"
PROJECT="/home/data/projets-aps/projet6/new_model"
DATA="/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet/Ewe_parquet/data_ewe"

export HF_HOME="/home/data/projets-aps/projet6/hf_cache"
export HF_DATASETS_CACHE="/home/data/projets-aps/projet6/hf_cache/datasets"
export TRANSFORMERS_CACHE="/home/data/projets-aps/projet6/hf_cache"
export TMPDIR="/home/data/projets-aps/projet6/tmp"
export PYTHONUNBUFFERED=1

cd "$PROJECT"
mkdir -p logs

echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Checkpoint: checkpoints/elus_ewe_v3_best.pt"
echo "========================================="

"$PYTHON" test_best.py \
    --checkpoint checkpoints/elus_ewe_v3_best.pt \
    --vocab checkpoints/symbols.json \
    --parquet_dir "$DATA" \
    --device cuda \
    --eval_batches 30 \
    --max_audio_sec 15.0

echo "========================================="
echo "Test terminé."
