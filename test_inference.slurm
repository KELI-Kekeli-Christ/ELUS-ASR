#!/bin/bash
#SBATCH --job-name=elus-test-inf
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/test_inference_%j.out
#SBATCH --error=logs/test_inference_%j.err

set -euo pipefail

PYTHON="/home/data/projets-aps/projet6/env_aps/bin/python3"
PROJECT="/home/data/projets-aps/projet6/new_model"
DATA="/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet/Ewe_parquet/data_ewe"

export HF_HOME="/home/data/projets-aps/projet6/hf_cache"
export TMPDIR="/home/data/projets-aps/projet6/tmp"
export PYTHONUNBUFFERED=1

cd "$PROJECT"
mkdir -p logs

CKPT="${1:-checkpoints/elus_ewe_v3_best.pt}"
echo "Test inférence : $CKPT (CPU — évite segfault GPU sur audio réel)"

"$PYTHON" test_inference.py \
    --checkpoint "$CKPT" \
    --vocab checkpoints/symbols.json \
    --parquet_dir "$DATA" \
    --device cpu \
    --pool_size 30 \
    --n_demo 6 \
    --max_audio_sec 15.0

echo "Test terminé."
