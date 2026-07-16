#!/bin/bash
# Télécharge LibriSpeech train-clean-100 + dev-clean + test-clean ([SLR12](https://www.openslr.org/12))
set -euo pipefail

DATA_ROOT="${1:-/home/data/projets-aps/projet6/data/librispeech}"
BASE_URL="https://openslr.trmal.net/resources/12"
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

download_and_extract() {
    local name="$1"
    if [ -d "LibriSpeech/${name}" ] || [ -d "${name}" ]; then
        echo "Déjà présent : ${name}"
        return 0
    fi
    echo "Téléchargement ${name}.tar.gz ..."
    wget -c "${BASE_URL}/${name}.tar.gz" -O "${name}.tar.gz"
    echo "Extraction ${name} ..."
    tar -xzf "${name}.tar.gz"
    rm -f "${name}.tar.gz"
}

download_and_extract "train-clean-100"
download_and_extract "dev-clean"
download_and_extract "test-clean"

echo "LibriSpeech prêt sous $DATA_ROOT"
find "$DATA_ROOT" -maxdepth 2 -type d | head -10
du -sh "$DATA_ROOT"/LibriSpeech/* 2>/dev/null || du -sh "$DATA_ROOT"/*
