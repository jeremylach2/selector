#!/usr/bin/env bash
# Runs the WSL2 half of Step 9 (Essentia pretrained classifiers) from
# Windows in one command:
#   wsl -d Ubuntu -- bash /mnt/d/codeprojects/spotifyProject/scripts/run_essentia.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ~/essentia_venv/bin/activate
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
python3 -m selector.audio.features_essentia "$@"
