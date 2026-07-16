#!/bin/bash
#SBATCH --job-name=elus-ewe-v4b
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail

PYTHON="/home/data/projets-aps/projet6/env_aps/bin/python3"
PROJECT="/home/data/projets-aps/projet6/new_model"
DATA="/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet/Ewe_parquet/data_ewe"
ALIGN="alignments/train.jsonl"
VOCAB="checkpoints/symbols_v4b.json"

export HF_HOME="/home/data/projets-aps/projet6/hf_cache"
export TMPDIR="/home/data/projets-aps/projet6/tmp"
export PYTHONUNBUFFERED=1

cd "$PROJECT"
mkdir -p logs checkpoints alignments

echo "=== ELUS v4b : segmental chars + alignements forcés + DTW decode ==="

N_ALIGN=$([ -f "$ALIGN" ] && wc -l < "$ALIGN" || echo 0)
if [ "$N_ALIGN" -lt 25000 ]; then
  echo "Génération des alignements forcés ($N_ALIGN lignes existantes, besoin ~29912)..."
  rm -f "$ALIGN"
  "$PYTHON" build_alignments.py \
      --parquet_dir "$DATA" \
      --split train \
      --out "$ALIGN" \
      --max_segmental_frames 128
else
  echo "Alignements OK : $N_ALIGN lignes"
fi

echo "Entraînement v4b..."
"$PYTHON" train.py \
    --parquet_dir "$DATA" \
    --split train \
    --rebuild_vocab \
    --vocab_out "$VOCAB" \
    --token_mode words \
    --segmental_on_chars \
    --segmental_weight 0.6 \
    --max_segmental_frames 128 \
    --alignments "$ALIGN" \
    --batch_size 8 \
    --grad_accum 4 \
    --grad_clip 1.0 \
    --early_stop_patience 5 \
    --epochs 20 \
    --num_workers 0 \
    --eval_batches 30 \
    --save_path checkpoints/elus_ewe_v4b.pt \
    2>&1 | tee "logs/train_${SLURM_JOB_ID:-local}.log"
