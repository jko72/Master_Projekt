#!/usr/bin/env bash
# Usage: bash submit_validation_diagnosis.sh <config> [diagnostic arguments]
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export HPC_ROOT="${HPC_ROOT:-$(dirname -- "$PROJECT")}"
HPC_ROOT="$(cd -- "$HPC_ROOT" && pwd -P)"
CONFIG="${1:?Usage: bash submit_validation_diagnosis.sh <config> [diagnostic arguments]}"
shift

[[ -f "$PROJECT/$CONFIG" ]] || { echo "Config fehlt: $CONFIG" >&2; exit 1; }
mkdir -p -- "$HPC_ROOT/logs/slurm"

sbatch \
  --chdir="$PROJECT" \
  --job-name="val-data-$(basename -- "$CONFIG" .yaml)" \
  --output="$HPC_ROOT/logs/slurm/%x-%j.out" \
  --error="$HPC_ROOT/logs/slurm/%x-%j.err" \
  "$PROJECT/diagnose_temporal_mos_validation.slurm" "$CONFIG" "$@"
