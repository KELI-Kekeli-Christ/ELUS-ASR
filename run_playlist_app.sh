#!/bin/bash
# Lance l'app playlist sur le serveur (port 7860)
set -euo pipefail

PYTHON="/home/data/projets-aps/projet6/env_aps/bin/python3"
PROJECT="/home/data/projets-aps/projet6/new_model"
export HF_HOME="/home/data/projets-aps/projet6/hf_cache"
export HUGGINGFACE_HUB_CACHE="/home/data/projets-aps/projet6/hf_cache/hub"
export PYTHONUNBUFFERED=1

cd "$PROJECT"

PLAYLIST="${1:-$PROJECT/demo_playlist}"
mkdir -p "$PLAYLIST"

# Démo : quelques wav LibriSpeech test-clean si playlist vide
if [ -z "$(ls -A "$PLAYLIST" 2>/dev/null)" ]; then
  echo "Playlist vide → copie de 20 clips test-clean LibriSpeech comme démo"
  LS="/home/data/projets-aps/projet6/data/librispeech/LibriSpeech/test-clean"
  i=0
  find "$LS" -name "*.flac" | head -20 | while read -r f; do
    base=$(basename "$f" .flac)
    # paroles = transcription LibriSpeech
    chap=$(dirname "$f")
    uid="$base"
    grep "^$uid " "$chap"/*.trans.txt 2>/dev/null | head -1 | cut -d' ' -f2- > "$PLAYLIST/${base}.txt" || true
    cp "$f" "$PLAYLIST/${base}.flac"
  done
fi

"$PYTHON" -c "import gradio" 2>/dev/null || "$PYTHON" -m pip install -q gradio

exec "$PYTHON" app_playlist.py \
  --checkpoint checkpoints/elus_librispeech_100h_best.pt \
  --vocab checkpoints/symbols_en.json \
  --playlist "$PLAYLIST" \
  --device cpu \
  --server_name 0.0.0.0 \
  --server_port 7860 --share
