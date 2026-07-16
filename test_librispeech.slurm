#!/bin/bash
#SBATCH --job-name=elus-ls-test
#SBATCH --partition=gpu
#SBATCH --nodelist=achille
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/test_ls_%j.out
#SBATCH --error=logs/test_ls_%j.err

set -euo pipefail

PYTHON="/home/data/projets-aps/projet6/env_aps/bin/python3"
PROJECT="/home/data/projets-aps/projet6/new_model"
LS="/home/data/projets-aps/projet6/data/librispeech"
CKPT="${1:-checkpoints/elus_librispeech_100h_best.pt}"

export HF_HOME="/home/data/projets-aps/projet6/hf_cache"
export HUGGINGFACE_HUB_CACHE="/home/data/projets-aps/projet6/hf_cache/hub"
export TMPDIR="/home/data/projets-aps/projet6/tmp"
export PYTHONUNBUFFERED=1

cd "$PROJECT"
mkdir -p logs

echo "=== Test texte → audio LibriSpeech test-clean ==="
echo "Checkpoint: $CKPT"
echo "Node: $(hostname)"

"$PYTHON" test_inference.py \
    --checkpoint "$CKPT" \
    --vocab checkpoints/symbols_en.json \
    --librispeech_dir "$LS" \
    --split test-clean \
    --device cpu \
    --pool_size 50 \
    --n_demo 10 \
    --max_audio_sec 20.0

echo "Test terminé."

