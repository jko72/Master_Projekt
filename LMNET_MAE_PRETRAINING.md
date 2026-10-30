# Paralleles LMNet-MAE-Pretraining auf p1

Die neue Variante wird ausschließlich mit
`src/configs/semanticKitti_MAE_Pretrain_temporal_e2e_lmnet_p1_2gpu.yaml`
aktiviert. Die bisherigen MAE-Configs, ihre Legacy-Projektion, ihre
Augmentation und ihr Checkpointformat v1 bleiben verwendbar. Die frühere
vierkanalige T-MAE-Pipeline bleibt ebenfalls unverändert.

## Neues Profil

| Einstellung | Wert |
| --- | --- |
| Auflösung und FoV | 64×2048, +3°/−25° |
| Eingang | `[x,y,z,range,intensity]`, Historie `[8]`, 10 Hz |
| Projektion | LMNet floor/Clamp, Index0 gültig, finite/r>0, keine obere Rangegrenze |
| Normalisierung | `paper_normalization`, wie beim LMNet-Temporal-MOS |
| Augmentation | Gemeinsame LMNet-Spiegelung/Translation nach Egomotion; Dropout nur aktuell; nur im Training |
| Fehlende Historie | Nullfeatures, valid=false, nominales negatives dt; Sequenzanfänge bleiben erhalten |
| Rekonstruktion | Fünf standardisierte XYZRI-Kanäle; Target vor MAE-Maskierung sichern |
| Maske | Patch 8×64, mask_ratio=.5, nur gültige aktuelle Pixel |
| Optimierung | AdamW .0005, WeightDecay .0001, Warmup-Cosine mit Warmup2, 50 Epochen |
| p1/P100 | Zwei GPUs, FP32 (`amp=false`) |
| Effektive Globalbatch | 1 Microbatch × 2 GPUs × 8 Akkumulationsschritte = 16 |

Die 50 MAE-Epochen und die Lossgewichte xyz=.5, range=1, intensity=.25,
Smooth-L1-beta=1 stammen aus der bestehenden MAE-Config. Sie sind keine
Autorenregeln von LMNet. Der überwachte MOS-Lauf behält seine eigene
100-Epochen-/SGD-Konfiguration.

Die Maskenproportionen sind exakt erhalten:
8/64 in der Höhe und 64/2048=16/512 in der Breite. Beide Bildgrößen haben
8×32=256 mögliche Patches. Die tatsächlich maskierte gültige Pixelzahl kann
wegen ganzer Patches leicht über 50 % liegen.

Es werden weder semantische Rohlabels noch Label-/Residualcaches für MAE
gelesen. Rekonstruktionsziele entstehen online aus den Scans. Die offiziellen
Testsequenzen 11–21 bleiben vom Pretraining ausgeschlossen. Val08 wird
vollständig ohne Augmentation und mit reproduzierbarer MAE-Maske verwendet.

## Akkumulation und Checkpoints

Microbatch1 wurde gewählt, damit nicht acht Fullres-Samples gleichzeitig auf
einer p1-GPU liegen müssen. Pro Akkumulationsgruppe werden CPU-seitig die
gültigen maskierten Zielwerte gezählt. Die Gradienten entsprechen der über
diese Gruppe und beide DDP-Ranks gemittelten komponentenweisen Pixel-Loss.
DDP synchronisiert erst am letzten Microbatch. Auch eine unvollständige
Restgruppe wird anhand ihrer tatsächlichen Zielzahl normiert.

BatchNorm und Dropout arbeiten weiterhin pro Microbatch. Akkumulation ist
deshalb keine identische Nachbildung einer physischen Batchgröße8 pro GPU.
Es gibt keine automatische LR-Skalierung. Globalbatch, Microbatch,
Akkumulation und Maskenpatch werden im Runmanifest gespeichert;
Optimizerupdates, AMP-Skips und tatsächliche Samplezahl stehen im Trainings-CSV.

LMNet-Checkpoints verwenden das zusätzliche Format
`temporal_e2e_masked_reconstruction_lmnet_v2`. Es speichert den expliziten
Projektions-/Filter-/Historienvertrag. Legacy-v1-Metadaten bleiben unverändert.
Beim LMNet-MOS-Transfer werden zusätzlich Projektion, Auflösung/FoV,
Filter, Normalisierung, Historienoffsets und Scanrate geprüft. Ein Legacy-
Checkpoint wird nicht still als LMNet-Pretraining interpretiert.
Alle Backbone-/Temporal-/Decodergewichte werden übertragen, lediglich der
Rekonstruktionskopf wird durch den binären MOS-Kopf ersetzt.

## Startbefehle

Aus dem Repository auf dem HPC; diese Befehle wurden nicht eingereicht:

```bash
cd /home/koehlbrandt/hpc/LiDARSensorForecaster

# Optional: kleiner p1-Realdaten-Smoke-Job, zwei GPUs und ein Optimizerupdate.
bash submit_temporal_e2e_mae_lmnet_p1_2gpu.sh \
  src/configs/semanticKitti_MAE_Pretrain_temporal_e2e_lmnet_p1_2gpu.yaml \
  --epochs 1 --max_batches 2 --num_workers 0

# Vollständiges Pretraining: 50 Epochen, effektive Globalbatch16.
bash submit_temporal_e2e_mae_lmnet_p1_2gpu.sh
```

Der neue Wrapper verwendet den bestehenden p1-/Apptainer-/DDP-Startpfad.
`HPC_ROOT` und `IMAGE` können wie bisher gesetzt werden. Daten werden
schreibgeschützt unter `/hpc/SemanticKitti/dataset/sequences` eingebunden.
Die neue Logablage liegt auf dem Host unter:

```text
<HPC_ROOT>/logs/temporal_e2e_mae_lmnet/
  semanticKitti_MAE_Pretrain_temporal_e2e_lmnet_p1_2gpu/job-<ID>/
    MAE_Pretrain_temporal_e2e_lmnet_fullres_p1_2gpu_global16_<Zeit>/
      checkpoints/{last.pt,best_val_loss.pt}
```

Für Resume muss der konkrete **Containerpfad** zu `last.pt` angegeben werden:

```bash
bash submit_temporal_e2e_mae_lmnet_p1_2gpu.sh \
  src/configs/semanticKitti_MAE_Pretrain_temporal_e2e_lmnet_p1_2gpu.yaml \
  --resume /hpc/logs/temporal_e2e_mae_lmnet/<Config>/job-<ID>/<Run>/checkpoints/last.pt
```

Das spätere MOS-Finetuning kann den bereits vorhandenen p2-LMNet-Startpfad
verwenden. Den konkreten Containerpfad einsetzen; ohne `--pretrained_path`
bleibt die bisherige MOS-Config ein Scratch-Lauf:

```bash
P2_GPU_GRES=gpu:2 bash submit_mos_p2_2gpu.sh \
  src/configs/mos_semanticKitti_temporal_e2e_lmnet_v1_p2_2gpu_online.yaml \
  --pretrained_path /hpc/logs/temporal_e2e_mae_lmnet/<Config>/job-<ID>/<Run>/checkpoints/best_val_loss.pt
```

## Ausgeführte Prüfungen, 29.10.2026

- Gemeinsame Suite: 93 Tests erfolgreich, zwei CUDA-Tests übersprungen.
  Log: `/tmp/lmnet-mae-full-regression.log`.
- Neue Tests prüfen labelunabhängiges Laden ohne Caches, MAE-/MOS-Parität,
  Randframes, synchrone Augmentation, Masken-/Targettrennung, p1-Launcher mit
  einem sbatch-Stub, Formate v1/v2, inkompatible Transfers und exaktes Resume.
- Akkumulationsgradienten wurden ohne BatchNorm gegen eine gemeinsame
  physische Batch geprüft, mit unterschiedlichen Maskenzahlen und Restgruppe;
  zusätzlich mit zwei lokalen CPU-DDP-Prozessen.
- Synthetischer Forward/Backward/AdamW-Schritt bei [1,2,5,64,2048]:
  Loss=.9105077, alle Modellgruppen mit endlichen aktiven Gradienten.
  Log: `/tmp/lmnet-mae-fullres-smoke.log`.
- Echter Fullres-AdamW-Schritt auf Val08/000008 mit eingefrorener
  Validierungsaugmentation: Loss=.9536660, 49.058 maskierte gültige Pixel,
  Maskenanteil≈50,17 %. Es wurden ausschließlich die zwei .bin-Scans für
  Frame0 und Frame8 gelesen; keine .label- oder .npy-Dateien.
  Benchmarkeingaben identisch, Frame0-Historie Null/ungültig.
  Ergebnis: `/tmp/lmnet-mae-real-smoke.json`.
- Configloader, Shellsyntax und `git diff --check` erfolgreich.

Die lokale Umgebung bietet nur CPU-PyTorch und keinen nutzbaren NVIDIA-
Treiber. Ein echter p1-GPU-/Speichertest sowie vollständiges Pretraining und
anschließendes MOS-Finetuning stehen noch aus. Es wurde kein HPC-Job und
kein vollständiges Training gestartet. Eine Leistungsverbesserung wird
nicht behauptet.
