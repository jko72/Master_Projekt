#!/usr/bin/env bash
# Aufruf: bash submit_mos.sh src/configs/mos_semanticKitti_temporal_e2e.yaml [Trainingsargumente]
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export HPC_ROOT="${HPC_ROOT:-$(dirname -- "$PROJECT")}"
HPC_ROOT="$(cd -- "$HPC_ROOT" && pwd -P)"
CONFIG="${1:-src/configs/mos_semanticKitti_temporal_e2e.yaml}"
if (( $# > 0 )); then
  shift
fi
[[ -f "$PROJECT/$CONFIG" ]] || { echo "Config fehlt (Pfad relativ zum Repository): $CONFIG" >&2; exit 1; }
CONFIG_NAME="$(basename -- "$CONFIG")"
CONFIG_NAME="${CONFIG_NAME%.*}"
mkdir -p -- "$HPC_ROOT/logs/slurm"

sbatch \
  --partition=p1 \
  --gres=gpu:2 \
  --chdir="$PROJECT" \
  --job-name="$CONFIG_NAME" \
  --output="$HPC_ROOT/logs/slurm/%x-%j.out" \
  --error="$HPC_ROOT/logs/slurm/%x-%j.err" \
  "$PROJECT/train_mos.slurm" "$CONFIG" "$@"
