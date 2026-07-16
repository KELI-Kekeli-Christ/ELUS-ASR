#!/bin/bash
# Cherche le Whisper medium fine-tuné éwé sur le serveur
ROOT="${1:-/home/data/projets-aps/projet6}"
echo "Recherche Whisper éwé sous $ROOT ..."
find "$ROOT" -maxdepth 5 -type f \( -name 'preprocessor_config.json' -o -name 'tokenizer_config.json' \) 2>/dev/null \
  | while read -r f; do
    d=$(dirname "$f")
    if [ -f "$d/config.json" ] && grep -qi whisper "$d/config.json" 2>/dev/null; then
      echo "CANDIDAT: $d"
      grep -E '"model_type"|"vocab_size"|"_name_or_path"' "$d/config.json" 2>/dev/null | head -3
    fi
  done
