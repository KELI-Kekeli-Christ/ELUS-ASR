#!/bin/bash
#SBATCH --job-name=elus-ls-en
#SBATCH --partition=gpu
#SBATCH --nodelist=achille
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_ls_%j.out
#SBATCH --error=logs/slurm_ls_%j.err

set -euo pipefail

PYTHON="/home/data/projets-aps/projet6/env_aps/bin/python3"
PROJECT="/home/data/projets-aps/projet6/new_model"
LS="/home/data/projets-aps/projet6/data/librispeech"
ALIGN="alignments/librispeech_train_words.jsonl"
VOCAB="checkpoints/symbols_en.json"

export HF_HOME="/home/data/projets-aps/projet6/hf_cache"
export HF_DATASETS_CACHE="/home/data/projets-aps/projet6/hf_cache/datasets"
export TRANSFORMERS_CACHE="/home/data/projets-aps/projet6/hf_cache"
export HUGGINGFACE_HUB_CACHE="/home/data/projets-aps/projet6/hf_cache/hub"
export TMPDIR="/home/data/projets-aps/projet6/tmp"
export PYTHONUNBUFFERED=1
# Contourne segfault CUDA init sur certains nœuds
export CUDA_MODULE_LOADING=LAZY

cd "$PROJECT"
mkdir -p logs checkpoints alignments

echo "=== ELUS LibriSpeech EN — segmental MOT par frame ==="
echo "Node: $(hostname)"

if [ ! -d "$LS/LibriSpeech/train-clean-100" ]; then
  echo "Téléchargement LibriSpeech..."
  bash download_librispeech.sh "$LS"
fi

if [ ! -f "$VOCAB" ]; then
  echo "Construction vocabulaire anglais..."
  "$PYTHON" -c "
from data import build_vocab_from_librispeech, save_vocab_meta
v = build_vocab_from_librispeech('$LS', 'train-clean-100')
save_vocab_meta(v, '$VOCAB')
print('Vocab:', len(v))
"
fi

N_ALIGN=$([ -f "$ALIGN" ] && wc -l < "$ALIGN" || echo 0)
if [ "$N_ALIGN" -lt 25000 ]; then
  echo "Alignements forcés MOTS ($N_ALIGN lignes)..."
  rm -f "$ALIGN"
  "$PYTHON" build_alignments.py \
      --librispeech_dir "$LS" \
      --split train-clean-100 \
      --vocab "$VOCAB" \
      --level words \
      --out "$ALIGN" \
      --max_segmental_frames 128
fi

echo "Entraînement ELUS anglais (HuBERT + segmental mots + alignements)..."
"$PYTHON" train.py \
    --librispeech_dir "$LS" \
    --split train-clean-100 \
    --val_split dev-clean \
    --vocab "$VOCAB" \
    --token_mode words \
    --audio_pretrained facebook/hubert-base-ls960 \
    --segmental_weight 0.6 \
    --max_segmental_frames 128 \
    --alignments "$ALIGN" \
    --batch_size 4 \
    --grad_accum 8 \
    --grad_clip 1.0 \
    --early_stop_patience 5 \
    --epochs 15 \
    --num_workers 0 \
    --eval_batches 20 \
    --save_path checkpoints/elus_librispeech_100h.pt \
    2>&1 | tee "logs/train_ls_${SLURM_JOB_ID:-local}.log"
