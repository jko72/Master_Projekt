# SalsaNext: Full-Resolution-Residualvergleich auf P2

Beide Versuche trainieren `salsanext_mos` von Grund auf mit sechs Kanälen
`[x,y,z,range,intensity,residual_1]` auf 64×2048. Die vorhandene Projektion
und der Range-Filter bleiben bestehen. Die ersten fünf Kanäle verwenden
`paper_normalization`; der Residualkanal bleibt unverändert.

| Einstellung | Versuch A | Versuch B |
|---|---|---|
| Config in `src/configs/` | `mos_semanticKitti_residual1_fullres_lovasz_w10_p2_2gpu.yaml` | `mos_semanticKitti_residual1_fullres_lovasz_w290_p2_2gpu.yaml` |
| Static / Moving | 1 / 10 | 1 / 290.34 |

Alle anderen P2-Trainingseinstellungen sind identisch: Seed 42, 150 Epochen,
AdamW mit LR 0.0005 und Weight Decay 0.0001, Cosine bis 0.000005, Dropout 0.1,
Gradient Clipping 5, kein Vortraining, AMP aus. `batch_size: 4` gilt **pro GPU**;
zwei DDP-Prozesse ergeben globale Batchgröße 8, ohne Gradient Accumulation.
Training: 00–07 und 09–10; Validierung: 08.

Die Configs lesen unter `/hpc/SemanticKitti/dataset/sequences/XX/` aus
`residual_images_1_Full_Res/` und `mos_labels_2048/`. Der Container bindet den
HPC-Arbeitsordner nach `/hpc`. `strict_residuals: true` macht fehlende,
falsch dimensionierte oder nicht endliche Residuals zum Fehler. Bestehende
Configs behalten den bisherigen Warnungs-/Null-Fallback.

## Start auf dem HPC-Login-Knoten

```bash
cd /home/koehlbrandt/hpc/LiDARSensorForecaster
bash submit_mos_residual_comparison_p2.sh
```

Das Skript reicht vier Jobs ein, jeder mit zwei GPUs auf Partition `p2`:
zwei kurze GPU-Smoke-Tests und beide vollständigen Trainings. Beide Trainings
haben `afterok`-Abhängigkeiten von **beiden** Smoke-Tests. Damit laufen die
Trainings erst nach erfolgreichen Tests. Job-IDs werden ausgegeben. Nach einem
fehlgeschlagenen Smoke-Test werden die abhängigen Trainings nicht gestartet.

Ein einzelner Lauf lässt sich so einreichen:

```bash
bash submit_mos_residual_p2_2gpu.sh \
  src/configs/mos_semanticKitti_residual1_fullres_lovasz_w10_p2_2gpu.yaml
```

Für einen einzelnen GPU-Test `--smoke_test` anhängen. Dieser verwendet die
ersten 16 Trainingsframes und 9 Validierungsframes bei zwei GPUs und Batch 4,
trainiert eine Epoche und schreibt in einen separaten `*_SMOKE`-Lauf. Er prüft
den echten Full-Resolution-Forward-/Backward-Pfad und die ungleich großen
Validierungsshards, liefert aber keinen Benchmark-Score.

GPU-Kurztests und Trainings fordern ausdrücklich `gpu:a100_40gb:2` an,
also zwei A100 mit jeweils 40 GB. Dies gilt auch bei direktem `sbatch` auf
das neue Slurm-Skript. Abweichende Werte für `P2_GPU_GRES` werden abgelehnt.
Die vorhandenen temporalen P2-Skripte bleiben unverändert.

## P1 mit zwei 16-GB-GPUs und AMP

Für P1 gibt es separate Configs mit dem Suffix `_p1_2gpu_amp.yaml`, jeweils
für `w10` und `w290`. Sie unterscheiden sich von den entsprechenden P2-Configs
nur durch `amp: true`, `amp_dtype: float16` und den Laufnamen. Batchgröße 4 pro
GPU, globale Batchgröße 8, Full Resolution und 150 Epochen bleiben erhalten.
P1 verwendet wie das vorhandene P1-Skript `--partition=p1 --gres=gpu:2`.

Nur die beiden GPU-Tests einreichen:

```bash
cd /home/koehlbrandt/hpc/LiDARSensorForecaster
bash submit_mos_residual_p1_2gpu.sh \
  src/configs/mos_semanticKitti_residual1_fullres_lovasz_w10_p1_2gpu_amp.yaml \
  --smoke_test
bash submit_mos_residual_p1_2gpu.sh \
  src/configs/mos_semanticKitti_residual1_fullres_lovasz_w290_p1_2gpu_amp.yaml \
  --smoke_test
```

Dies reicht keine vollständigen Trainings ein. Nach erfolgreichem GPU-Test
lässt sich mit demselben Einzelaufruf ohne `--smoke_test` das jeweilige Training
auf P1 starten. Der tatsächliche Speicherbedarf auf den 16-GB-GPUs wird erst
durch den GPU-Test geprüft; in der Bearbeitungsumgebung ist keine GPU verfügbar.

Der Trainer verwendet Autocast und einen über Epochen erhaltenen GradScaler
(FP16-Startskala 128), entpackt die Gradienten vor Clipping und überspringt
Updates bei FP16-Gradientenüberlauf. Lovász sowie die CE-Normalisierung bleiben
in FP32. Das ist insbesondere bei hohen Klassengewichten nötig, da die Summe
der Gewichte den FP16-Wertebereich überschreiten kann. Der Scaler-Zustand wird
im Checkpoint gespeichert. Logs und CSV zeigen erfolgreiche Optimizer-Schritte,
übersprungene AMP-Schritte, die Skala und den maximalen GPU-Speicherverbrauch
über beide Prozesse. Ein Smoke-Test ohne erfolgreichen Optimizer-Schritt gilt
als fehlgeschlagen.

## Loss, DDP und Ausgaben

`tools/train_mos.py` unterstützt nun optional `mos_train_params.loss`:

```yaml
loss:
  type: ce_lovasz
  lovasz_weight: 1.0
  lovasz_classes: present
  lovasz_per_image: false
```

Ohne diesen Eintrag bleibt der Loss CE allein. Die vorhandene Implementierung
aus `src/helper/mos_lovasz_loss.py` wird wiederverwendet. CE-Klassengewichte
gelten ausschließlich für CE; Lovász bekommt Softmax-Wahrscheinlichkeiten in
FP32. Ignore bleibt -1, Static 0, Moving 1.

DDP korrigiert die lokale gewichtete CE so, dass ihr Gradient der CE auf der
globalen Batch entspricht. Lovász sortiert pro GPU und wird über die beiden
lokalen Objectives gemittelt; dies entspricht nicht Lovász auf einer gemeinsam
sortierten globalen Batch. BatchNorm bleibt gewöhnliche BatchNorm pro GPU.
Vor der Validierung werden die laufenden Statistiken von Rank 0 übernommen.
Jeder Validierungsframe wird exakt einmal gezählt, ohne Padding-Duplikate.
Trainingssampling verwendet den üblichen DistributedSampler mit gleicher
Schrittzahl pro Rank und gegebenenfalls Padding am Epochenende.

Nur Rank 0 schreibt TensorBoard, Config, CSV und Checkpoints. `metrics.csv`
enthält `train_loss_ce`, `train_loss_lovasz`, `train_loss_total` und die
entsprechenden `val_*`-Spalten. CE wird über die Summe der gültigen
Klassengewichte aggregiert, Lovász über die lokalen Batchwerte. Die neuen
Gesamt-Loss-Werte sind deren Summe. Alte Single-Process-CE-Läufe behalten ihre
bisherige samplegewichtete `train_loss`-/`val_loss`-Ausgabe.

Ausgaben befinden sich unter
`logs/mos/<Configname>/job-<Job-ID>/<Laufname>_<Zeitstempel>/`.
`best_moving_iou.pt` wird anhand der Pixel-Moving-IoU auf Sequenz 08 gewählt;
`last.pt` wird alle 15 Epochen und zum Schluss geschrieben. Die Checkpoints
enthalten unpräfixierte `model_state_dict`-Schlüssel und die Config unter `cfg`
wie bisher. Der endgültige Punkt-Score wird mit dem externen Evaluator bestimmt.

## Prüfung am 26.10.2026

- Gesamte Testsuite: 42 Tests, davon 41 erfolgreich und ein CUDA-Test mangels
  GPU übersprungen. Darunter neue Zwei-Prozess-Gloo-Tests gegen ein explizites
  Referenzobjective, ungleich große/leere Validierungsshards, zwei vollständige
  DDP-Epochen mit Checkpoints und CSV, Residualfehler und simulierte Slurm-Jobs.
- 22 echte Eingaben (Frames 000000 und 000100 in 00–10): sechs Kanäle,
  64×2048, endliche Werte, unveränderter Residualkanal und Übereinstimmung der
  gespeicherten Labels mit der vorhandenen Punkt-/Labelprojektion bestätigt.
- Python- und Bash-Syntax sowie `git diff --check` geprüft.
- In der Bearbeitungsumgebung fehlen Slurm/Apptainer und eine nutzbare GPU.
  Es wurde deshalb kein echter Slurm-Job eingereicht und kein GPU-Speicherbedarf
  gemessen. Die vorgeschalteten P2-Smoke-Tests prüfen dies vor den Trainings.
