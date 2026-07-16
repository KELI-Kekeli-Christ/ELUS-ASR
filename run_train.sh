#!/bin/bash
# Lance l'entraînement ELUS sur le serveur joyeux (corpus éwé + AfriHuBERT).
set -euo pipefail

PYTHON="/home/data/projets-aps/projet6/env_aps/bin/python3"
PROJECT="/home/data/projets-aps/projet6/new_model"
DATA="/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet/Ewe_parquet/data_ewe"

export HF_HOME="/home/data/projets-aps/projet6/hf_cache"
export HF_DATASETS_CACHE="/home/data/projets-aps/projet6/hf_cache/datasets"
export TRANSFORMERS_CACHE="/home/data/projets-aps/projet6/hf_cache"
export TMPDIR="/home/data/projets-aps/projet6/tmp"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"
mkdir -p "$PROJECT/checkpoints" "$PROJECT/logs"
cd "$PROJECT"

MODE="${1:-smoke}"

if [[ "$MODE" == "smoke" ]]; then
    echo "=== Smoke test (60 secondes) ==="
    VOCAB_ARGS=()
    if [[ -f "$PROJECT/checkpoints/symbols.json" ]]; then
        VOCAB_ARGS=(--vocab "$PROJECT/checkpoints/symbols.json")
        echo "Vocabulaire existant réutilisé."
    fi
    timeout 60 "$PYTHON" train.py \
        --parquet_dir "$DATA" \
        --split train \
        --max_files 1 \
        --batch_size 2 \
        --epochs 1 \
        --max_steps 1 \
        --num_workers 0 \
        "${VOCAB_ARGS[@]}" \
        --save_path checkpoints/elus_smoke.pt \
        2>&1 | tee logs/smoke_test.log
    echo "Smoke test terminé (code: ${PIPESTATUS[0]})"
elif [[ "$MODE" == "full" ]]; then
    echo "=== Entraînement complet ==="
    "$PYTHON" train.py \
        --parquet_dir "$DATA" \
        --split train \
        --batch_size 8 \
        --epochs 20 \
        --num_workers 4 \
        --save_path checkpoints/elus_ewe.pt \
        2>&1 | tee logs/train_full.log
else
    echo "Usage: $0 [smoke|full]"
    exit 1
fi
