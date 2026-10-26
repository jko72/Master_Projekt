#!/usr/bin/env bash
# Aufruf: bash submit_mos_p2_2gpu.sh <config> [Trainingsargumente]
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
P2_GPU_GRES="${P2_GPU_GRES:-gpu:2}"
[[ "$P2_GPU_GRES" == gpu:* ]] || {
  echo "P2_GPU_GRES muss ein Slurm-GRES-Ausdruck sein, z. B. gpu:a100_40gb:2." >&2
  exit 1
}
mkdir -p -- "$HPC_ROOT/logs/slurm"

sbatch \
  --partition=p2 \
  --gres="$P2_GPU_GRES" \
  --chdir="$PROJECT" \
  --job-name="$CONFIG_NAME" \
  --output="$HPC_ROOT/logs/slurm/%x-%j.out" \
  --error="$HPC_ROOT/logs/slurm/%x-%j.err" \
  "$PROJECT/train_mos_p2_2gpu.slurm" "$CONFIG" "$@"
