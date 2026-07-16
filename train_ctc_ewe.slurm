#!/bin/bash
#SBATCH --job-name=elus-ctc-ewe
#SBATCH --partition=gpu
#SBATCH --nodelist=achille
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm_ctc_%j.out
#SBATCH --error=logs/slurm_ctc_%j.err

set -euo pipefail

PYTHON="/home/data/projets-aps/projet6/env_aps/bin/python3"
PROJECT="/home/data/projets-aps/projet6/new_model"
DATA="/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet/Ewe_parquet/data_ewe"

export HF_HOME="/home/data/projets-aps/projet6/hf_cache"
export HF_DATASETS_CACHE="/home/data/projets-aps/projet6/hf_cache/datasets"
export TRANSFORMERS_CACHE="/home/data/projets-aps/projet6/hf_cache"
export HUGGINGFACE_HUB_CACHE="/home/data/projets-aps/projet6/hf_cache/hub"
export TMPDIR="/home/data/projets-aps/projet6/tmp"
export PYTHONUNBUFFERED=1
export CUDA_MODULE_LOADING=LAZY

cd "$PROJECT"
mkdir -p logs checkpoints

echo "=== ELUS Ewe + CTC head ==="
echo "Node: $(hostname)"

# Phase 1 : tête CTC seule (encodeur gelé)
"$PYTHON" train_ctc.py \
    --elus_checkpoint checkpoints/elus_ewe_v3_best.pt \
    --vocab checkpoints/symbols.json \
    --parquet_dir "$DATA" \
    --train_mode head \
    --epochs 8 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 3e-4 \
    --grad_clip 1.0 \
    --early_stop_patience 4 \
    --eval_batches 30 \
    --max_audio_sec 20 \
    --device cuda \
    --save_path checkpoints/elus_ewe_ctc_head.pt \
    2>&1 | tee "logs/train_ctc_head_${SLURM_JOB_ID:-local}.log"

# Phase 2 : fine-tune dernières couches + tête
BEST="checkpoints/elus_ewe_ctc_head_best.pt"
if [ ! -f "$BEST" ]; then BEST="checkpoints/elus_ewe_ctc_head.pt"; fi
echo "Phase 2 last4 depuis $BEST"

"$PYTHON" train_ctc.py \
    --elus_checkpoint "$BEST" \
    --vocab checkpoints/symbols.json \
    --parquet_dir "$DATA" \
    --train_mode last4 \
    --epochs 10 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 5e-5 \
    --grad_clip 1.0 \
    --early_stop_patience 4 \
    --eval_batches 30 \
    --max_audio_sec 20 \
    --device cuda \
    --save_path checkpoints/elus_ewe_ctc.pt \
    2>&1 | tee "logs/train_ctc_last4_${SLURM_JOB_ID:-local}.log"

FINAL="checkpoints/elus_ewe_ctc_best.pt"
if [ ! -f "$FINAL" ]; then FINAL="checkpoints/elus_ewe_ctc.pt"; fi

"$PYTHON" eval_ctc.py \
    --checkpoint "$FINAL" \
    --vocab checkpoints/symbols.json \
    --parquet_dir "$DATA" \
    --split test \
    --device cpu \
    --max_samples 200 \
    --out "logs/ctc_test_${SLURM_JOB_ID}.json" \
    2>&1 | tee -a "logs/train_ctc_last4_${SLURM_JOB_ID:-local}.log"

echo "Terminé."
