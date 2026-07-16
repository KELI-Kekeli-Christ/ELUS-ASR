#!/bin/bash
#SBATCH --job-name=elus-hybrid
#SBATCH --partition=gpu
#SBATCH --nodelist=achille
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/hybrid_%j.out
#SBATCH --error=logs/hybrid_%j.err

set -euo pipefail

PYTHON="/home/data/projets-aps/projet6/env_aps/bin/python3"
PROJECT="/home/data/projets-aps/projet6/new_model"
DATA="/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet/Ewe_parquet/data_ewe"
WHISPER="${WHISPER_MODEL:-/home/data/projets-aps/projet6/data_preprocessing/pre_proccessing_ewe_whisper/whisper_medium_ewe_best}"
PROCESSOR="${WHISPER_PROCESSOR:-/home/data/projets-aps/projet6/data_preprocessing/pre_proccessing_ewe_whisper/whisper_medium_ewe_hf}"

export HF_HOME="/home/data/projets-aps/projet6/hf_cache"
export HF_DATASETS_CACHE="/home/data/projets-aps/projet6/hf_cache/datasets"
export TRANSFORMERS_CACHE="/home/data/projets-aps/projet6/hf_cache"
export HUGGINGFACE_HUB_CACHE="/home/data/projets-aps/projet6/hf_cache/hub"
export TMPDIR="/home/data/projets-aps/projet6/tmp"
export PYTHONUNBUFFERED=1

cd "$PROJECT"
mkdir -p logs

if [ -z "$WHISPER" ]; then
  echo "Recherche Whisper fine-tuné éwé..."
  bash find_whisper_ewe.sh /home/data/projets-aps/projet6 | tee logs/whisper_candidates.log
  WHISPER=$(grep -m1 '^CANDIDAT:' logs/whisper_candidates.log | sed 's/CANDIDAT: //')
  # chemins connus
  for p in \
    /home/data/projets-aps/projet6/whisper_medium_ewe \
    /home/data/projets-aps/projet6/fine_tuning_whisper_ewe \
    /home/data/projets-aps/projet6/whisper-ewe-medium \
    /home/data/projets-aps/projet6/models/whisper-medium-ewe; do
    if [ -f "$p/config.json" ]; then WHISPER="$p"; break; fi
  done
fi

if [ -z "$WHISPER" ] || [ ! -f "$WHISPER/config.json" ]; then
  echo "ERREUR: modèle Whisper introuvable. Export WHISPER_MODEL=/chemin/vers/model"
  exit 1
fi

echo "Whisper: $WHISPER"
echo "Node: $(hostname)"

# Test complet (1452 utt) — ~2-3h CPU/GPU; max_samples pour smoke via env
MAX="${MAX_SAMPLES:-0}"
ELUS_DEV="${ELUS_DEVICE:-cpu}"
WHISPER_DEV="${WHISPER_DEVICE:-cpu}"

"$PYTHON" hybrid_asr.py \
    --whisper_model "$WHISPER" \
    --whisper_processor "$PROCESSOR" \
    --elus_checkpoint checkpoints/elus_ewe_v3_best.pt \
    --elus_vocab checkpoints/symbols.json \
    --parquet_dir "$DATA" \
    --split test \
    --device cpu \
    --whisper_device "$WHISPER_DEV" \
    --elus_device "$ELUS_DEV" \
    --max_samples "$MAX" \
    --num_beams 5 \
    --n_best 5 \
    --alpha 0.2 \
    --beta 1.0 \
    --out "logs/hybrid_wer_${SLURM_JOB_ID}.json"

echo "Terminé."
