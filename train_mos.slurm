#!/usr/bin/env bash
#SBATCH --job-name=mos-temporal
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Auf dem HPC aus dem Repository-Verzeichnis starten:
#   cd /home_cu/<username>/hpc/LiDARSensorForecaster
#   sbatch train_mos.slurm
# Empfohlen fuer mehrere Configs und zugeordnete Slurm-Logs:
#   bash submit_mos.sh src/configs/mos_semanticKitti_temporal_e2e.yaml
# Erwartete Struktur im uebergeordneten HPC-Arbeitsordner:
#   LiDARSensorForecaster/ (Code, Configs und SIF-Image)
#   SemanticKitti/dataset/sequences/
#   logs/mos/
# Bei anderer Struktur vor sbatch HPC_ROOT und ggf. IMAGE setzen.
# Das v2-SIF wird beim ersten Lauf aus der Registry erstellt und danach wiederverwendet.
# Ein neuer latest-Tag wird nicht automatisch heruntergeladen, solange das SIF existiert.
# Optional: sbatch train_mos.slurm src/configs/mos_semanticKitti_temporal_e2e_intensity.yaml
# Weitere Trainingsargumente: sbatch train_mos.slurm <config> --epochs 1 --max_batches 2
# Zwei P100-GPUs, ein torchrun-Prozess pro GPU; batch_size gilt pro GPU.
# batch_size: 8 ergibt bei gradient_accumulation: 1 eine globale Batchgroesse von 16.

HPC_ROOT="${HPC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}/..}"
HPC_ROOT="$(cd -- "$HPC_ROOT" && pwd -P)"
PROJECT="${HPC_ROOT}/LiDARSensorForecaster"
NUM_GPUS=2
# Explizites IPv4-Rendezvous fuer einen Knoten; kein localhost:0/--standalone.
# Verschiedene Slurm-Job-IDs ergeben normalerweise verschiedene Ports.
# Bei einer Portkollision: MOS_MASTER_PORT=29501 bash submit_mos.sh <config>
MOS_PORT="${MOS_MASTER_PORT:-$((15000 + ${SLURM_JOB_ID:-$$} % 40000))}"
[[ "$MOS_PORT" =~ ^[0-9]{1,5}$ ]] && (( 10#$MOS_PORT >= 1024 && 10#$MOS_PORT <= 65535 )) || {
  echo "Fehler: MOS_MASTER_PORT muss zwischen 1024 und 65535 liegen." >&2
  exit 1
}
IMAGE="${IMAGE:-${PROJECT}/kav_s210891_lidar_forecastingv2_latest.sif}"
IMAGE_URI="docker://iwloci-registry01.intern.th-ab.de:5000/kav_s210891_lidar_forecastingv2:latest"
CONFIG="${1:-src/configs/mos_semanticKitti_temporal_e2e.yaml}"
CONFIG_NAME="$(basename -- "$CONFIG")"
CONFIG_NAME="${CONFIG_NAME%.*}"
RUN_ID="${SLURM_JOB_ID:-manual-$(date +%Y%m%dT%H%M%S)-$$}"
RUN_LOG_ROOT="/hpc/logs/mos/${CONFIG_NAME}/job-${RUN_ID}"
if (( $# > 0 )); then
  shift
fi

command -v apptainer >/dev/null || { echo "Fehler: apptainer ist nicht verfuegbar." >&2; exit 1; }
[[ "${SLURM_JOB_PARTITION:-p1}" == p1 ]] || { echo "Fehler: Dieses Skript verwendet ausschliesslich Partition p1." >&2; exit 1; }
[[ -f "$PROJECT/$CONFIG" ]] || { echo "Fehler: Config fehlt (Pfad relativ zum Repository): $CONFIG" >&2; exit 1; }
[[ -d "$HPC_ROOT/SemanticKitti/dataset/sequences" ]] || { echo "Fehler: SemanticKITTI sequences fehlt unter $HPC_ROOT/SemanticKitti/dataset/sequences" >&2; exit 1; }
HOST_RUN_LOG_ROOT="$HPC_ROOT/logs/mos/${CONFIG_NAME}/job-${RUN_ID}"
mkdir -p -- "$HOST_RUN_LOG_ROOT"
{
  printf 'job_id=%s\nconfig=%s\nimage=%s\n' "$RUN_ID" "$CONFIG" "$IMAGE"
  printf 'training_arguments='
  printf '%q ' "$@"
  printf '\n'
} > "$HOST_RUN_LOG_ROOT/job-info.txt"

# Gleichzeitige Jobs warten auf denselben Pull. Die endgueltige Datei wird erst
# nach erfolgreicher Konvertierung angelegt; ein Abbruch hinterlaesst kein Cache-SIF.
mkdir -p -- "$(dirname -- "$IMAGE")"
(
  LOCK_DIR="${IMAGE}.lockdir"
  BUILD_DIR=""
  LOCK_OWNED=0
  cleanup_image_build() {
    if [[ -n "$BUILD_DIR" ]]; then
      rm -f -- "$BUILD_DIR/image.sif"
      rmdir -- "$BUILD_DIR"
    fi
    if (( LOCK_OWNED )); then
      rmdir -- "$LOCK_DIR"
    fi
  }
  trap cleanup_image_build EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  # mkdir ist atomar und benoetigt keine Dateisystem-Unterstuetzung fuer flock.
  WAIT_START=$SECONDS
  while [[ ! -s "$IMAGE" ]]; do
    if mkdir -- "$LOCK_DIR" 2>/dev/null; then
      LOCK_OWNED=1
      break
    fi
    if [[ ! -d "$LOCK_DIR" ]]; then
      echo "Fehler: Image-Sperrverzeichnis kann nicht erstellt werden: $LOCK_DIR" >&2
      exit 1
    fi
    if (( SECONDS - WAIT_START >= 3600 )); then
      echo "Fehler: Seit einer Stunde auf SIF-Erstellung gewartet. Sperre pruefen: $LOCK_DIR" >&2
      echo "Falls kein Job mehr ein Image erstellt, das leere Sperrverzeichnis mit rmdir entfernen." >&2
      exit 1
    fi
    echo "Warte auf SIF-Erstellung durch einen anderen Job: $IMAGE"
    sleep 10
  done
  if [[ -s "$IMAGE" ]]; then
    echo "Vorhandenes SIF wird verwendet: $IMAGE"
  else
    echo "SIF wird erstmals aus der Registry erstellt: $IMAGE_URI"
    BUILD_DIR="$(mktemp -d "${IMAGE}.build.XXXXXX")"
    apptainer pull "$BUILD_DIR/image.sif" "$IMAGE_URI"
    mv -- "$BUILD_DIR/image.sif" "$IMAGE"
  fi
)

export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_OMP_NUM_THREADS=2
export APPTAINERENV_USE_LIBUV=0
# Alle Prozesse laufen auf demselben Knoten. Netzwerk-Bootstrap lokal halten;
# GPU-Kommunikation ueber NCCL/P2P bleibt aktiviert.
export APPTAINERENV_NCCL_SOCKET_IFNAME=lo
export APPTAINERENV_GLOO_SOCKET_IFNAME=lo
export APPTAINERENV_NCCL_IB_DISABLE=1
# Die von Slurm zugeteilten GPUs beibehalten; keine fremden GPU-Indizes auswaehlen.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi

echo "Training startet: $(date --iso-8601=seconds)"
echo "HPC-Arbeitsordner: $HPC_ROOT"
echo "Image: $IMAGE"
echo "Config: $CONFIG"
echo "Slurm-Job-ID: $RUN_ID"
echo "Partition: p1; GPUs pro Training: $NUM_GPUS; Prozesse: $NUM_GPUS"
echo "DDP rendezvous: 127.0.0.1:$MOS_PORT (static)"
echo "Container-Netzwerk: isoliert (--net --network none), nur Loopback"
echo "Trainingslogs auf dem Host: $HOST_RUN_LOG_ROOT"

apptainer exec --nv \
  --net --network none \
  --bind "${HPC_ROOT}:/hpc" \
  --bind "${HPC_ROOT}/SemanticKitti/dataset/sequences:/hpc/SemanticKitti/dataset/sequences:ro" \
  --pwd /hpc/LiDARSensorForecaster \
  "${IMAGE}" \
  python3 -u -m torch.distributed.run \
    --rdzv-backend=static --rdzv-id="$RUN_ID" \
    --rdzv-conf=timeout=60 \
    --master-addr=127.0.0.1 --master-port="$MOS_PORT" \
    --nnodes=1 --node-rank=0 --nproc-per-node="$NUM_GPUS" \
    tools/train_temporal_mos_e2e.py \
    --cfg_path "$CONFIG" \
    --log_root "$RUN_LOG_ROOT" \
    --device cuda \
    "$@"

echo "Training beendet: $(date --iso-8601=seconds)"
