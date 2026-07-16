#!/bin/bash
###############################################################################
# SLURM — Entraînement ELUS v3 (stable: segmental_weight=0.15, grad clip, best ckpt)
###############################################################################
#SBATCH --job-name=elus-ewe-v3
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

export HF_HOME="/home/data/projets-aps/projet6/hf_cache"
export HF_DATASETS_CACHE="/home/data/projets-aps/projet6/hf_cache/datasets"
export TRANSFORMERS_CACHE="/home/data/projets-aps/projet6/hf_cache"
export TMPDIR="/home/data/projets-aps/projet6/tmp"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"
mkdir -p "$PROJECT/checkpoints" "$PROJECT/logs"
cd "$PROJECT"

echo "========================================="
echo "Job ID       : ${SLURM_JOB_ID:-unknown}"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python       : $($PYTHON --version 2>&1)"
echo "Version      : ELUS v3 (segmental=0.15, grad_clip=1.0, early_stop=5)"
echo "========================================="

"$PYTHON" train.py \
    --parquet_dir "$DATA" \
    --split train \
    --rebuild_vocab \
    --token_mode words \
    --batch_size 8 \
    --grad_accum 4 \
    --grad_clip 1.0 \
    --segmental_weight 0.15 \
    --early_stop_patience 5 \
    --lr 1e-4 \
    --warmup_steps 500 \
    --epochs 20 \
    --num_workers 0 \
    --eval_batches 30 \
    --vocab_out checkpoints/symbols.json \
    --save_path checkpoints/elus_ewe_v3.pt \
    2>&1 | tee "logs/train_${SLURM_JOB_ID:-local}.log"

echo "Entraînement terminé."
