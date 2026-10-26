#!/usr/bin/env bash
#SBATCH --job-name=mos-val-data
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-04:00:00

set -euo pipefail

HPC_ROOT="${HPC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}/..}"
HPC_ROOT="$(cd -- "$HPC_ROOT" && pwd -P)"
PROJECT="${HPC_ROOT}/LiDARSensorForecaster"
IMAGE="${IMAGE:-${PROJECT}/kav_s210891_lidar_forecastingv2_latest.sif}"
CONFIG="${1:?Usage: sbatch diagnose_temporal_mos_validation.slurm <config> [diagnostic arguments]}"
shift

[[ -f "$PROJECT/$CONFIG" ]] || { echo "Config fehlt: $CONFIG" >&2; exit 1; }
[[ -f "$IMAGE" ]] || { echo "SIF fehlt: $IMAGE" >&2; exit 1; }
[[ -d "$HPC_ROOT/SemanticKitti/dataset/sequences" ]] || { echo "SemanticKITTI fehlt." >&2; exit 1; }

export APPTAINERENV_PYTHONUNBUFFERED=1
echo "Validation-Diagnose startet: $(date --iso-8601=seconds)"
echo "Config: $CONFIG"

apptainer exec --nv --net --network none \
  --bind "${HPC_ROOT}:/hpc" \
  --bind "${HPC_ROOT}/SemanticKitti/dataset/sequences:/hpc/SemanticKitti/dataset/sequences:ro" \
  --pwd /hpc/LiDARSensorForecaster \
  "$IMAGE" \
  python3 -u tools/diagnose_temporal_mos_validation.py --cfg_path "$CONFIG" "$@"

echo "Validation-Diagnose beendet: $(date --iso-8601=seconds)"
