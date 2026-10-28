# Masked Temporal Reconstruction mit der Temporal-E2E-Architektur

## Lokale Baseline

Referenz ist ausdrücklich die lokale HPC-Konfiguration
`src/configs/mos_semanticKitti_temporal_e2e_intensity_off8_64x512_lovasz_w290_aug.yaml`.
Sie bleibt unverändert. Verwendet werden 64 × 512, historische Offsets `[8]`,
XYZRI, `intensity_only`, CE+Lovász und MOS-Klassengewichte `[1,290]`.
Die Beispiele `[5,1]` aus der Aufgabenbeschreibung ersetzen diesen Offset nicht.

Neue Konfigurationen:

- `src/configs/semanticKitti_temporal_e2e_mae_intensity_off8_64x512_aug.yaml`:
  50 Epochen Pretraining, Batch 8 pro GPU, AdamW, LR 0.0005 → 0.000005,
  Weight Decay 0.0001, 2 Warmup-Epochen, Clip 5, Seed 42, BF16-AMP.
- `src/configs/semanticKitti_temporal_e2e_mae_intensity_off8_64x512_aug_p1_2gpu_fp32.yaml`:
  gleiche MAE-Konfiguration, AMP deaktiviert für P1/P100 ohne natives BF16.
  `amp_dtype: float16` ist bei deaktiviertem AMP ohne Rechenwirkung.
- `src/configs/mos_semanticKitti_temporal_e2e_mae_intensity_off8_64x512_lovasz_w290_aug.yaml`:
  Fine-Tuning unter identischen Bedingungen zur lokalen MOS-Baseline; nur der
  Run-Name unterscheidet sich. Der konkrete MAE-Checkpoint wird über
  `mos_train_params.pretrained_path` oder CLI `--pretrained_path` angegeben.
  `null` behält absichtlich das vorhandene Scratch-Verhalten.

YAMLs und HPC-Shellskripte unterliegen den vorhandenen lokalen
`.git/info/exclude`-Regeln und bleiben lokal. Es wurde nichts gepusht.

## Architektur und Eingabe

`SalsaNextTemporalE2E` verarbeitet weiterhin `[B,T,5,H,W]` über den gemeinsamen
SalsaNext-Encoder, `LocalHistoricalAttention` und alle vier Decoder-UpBlocks.
Die State-Dict-Schlüssel bleiben erhalten. Alle Modellparameter werden gemeinsam
trainiert. Einzige aufgabenspezifische Parameter sind `decoder.logits.weight`
und `.bias`: fünf Rekonstruktionskanäle im MAE, zwei MOS-Logits im Fine-Tuning.
Die bisherigen vierkanaligen MOS-Eingaben bleiben ebenfalls unterstützt.
Das bisherige `RangeViewSiamWCA`-T-MAE und dessen Vierkanal-Loss bleiben erhalten.

Die gemeinsame `TemporalE2EInputMixin` implementiert für beide neuen/vorhandenen
E2E-Datasets identisch: Rohscan laden, historische Punkte mit
`solve(current_pose, historical_pose)` aus den bestehenden Velodyne-Posen
transformieren, bestehende Range-View-Projektion mit expliziter Occupancy und
Kollisionsauflösung, Augmentation und bestehende Normalisierung.
Kanalreihenfolge: `[x,y,z,range,intensity]`.

`TemporalE2EMAEDataset` liest weder MOS-Labels noch deren Statistik. Es benötigt
nur Scans und Posen/Kalibrierung und liefert ein Dictionary mit `x`, `valid`,
`dt`, `mask`, `target_current`, `meta`. Die MOS-Schnittstelle bleibt unverändert.
Es verwendet dieselbe absteigende Offset-Sortierung und negative Sekunden für
die Historie, mit `dt[-1] = 0`. Jede positive Offset-Liste wie `[5]`, `[5,1]`
oder `[5,3,1]` funktioniert ohne Änderung am Modell. Keine Offset-Zufallsauswahl.

Die Baseline-Augmentationen sind im MAE ebenfalls nur beim Training aktiv.
Das Target ist das vollständige aktuelle Frame **nach derselben Augmentation
und Normalisierung**, aber vor der MAE-Maskierung. Historische Frames erhalten
keine künstliche MAE-Maske; der aktivierte historische Punkt-Dropout der Baseline
bleibt als separate Train-only-Augmentation erhalten.

Train: `00,01,02,03,04,05,06,07,09,10`; Validation: `08`.
Testsequenzen werden nicht zum Pretraining geladen. Sowohl Loader als auch
Dataset blockieren den Test-Split; das Dataset lehnt konfigurierte Testsequenzen ab.

## Maskierung und Loss

```yaml
mae_params:
  mask:
    type: patch
    patch_h: 8
    patch_w: 16
    mask_ratio: 0.50
    mask_only_valid: true
    apply_to: current
  loss:
    xyz_weight: 0.50
    range_weight: 1.00
    intensity_weight: 0.25
    beta: 1.0
```

Zufällig permutierte nichtleere Patches werden ausgewählt, bis etwa die
gewünschte Anzahl gültiger aktueller Pixel erreicht ist. Die Rate kann um
höchstens einen Patch überschritten werden, insbesondere bei dünner Occupancy.
Ungültige Löcher werden bei `mask_only_valid` aus der Maske ausgeschlossen.
Alle fünf Kanäle werden gemeinsam genullt, ausschließlich im aktuellen Frame.
Die Sensorvalidität bleibt unverändert und steuert weiter die Attention.
Skip-Features stammen vollständig aus dem maskierten Input. Der Trainer reicht
niemals den separaten unmaskierten Target-Tensor an das Modell weiter.

Masken sind deterministisch aus Seed, Sample-Index und Trainingsepoche;
Validation verwendet feste Masken für vergleichbare Checkpoints. Die Baseline-
Augmentation verwendet weiterhin die existierenden Rank-/Worker-RNG-Seeds.

Loss-Maske: `original_current_valid AND mae_mask`.
XYZ wird über drei Kanäle und ausgewählte Pixel gemittelt, Range und normalisierte
Intensität jeweils über ihre ausgewählten Pixel. Smooth-L1 und Reduktionen
laufen in FP32, auch bei AMP. Nichtendliche Werte werden vor der Loss-Arithmetik
maskiert; leere Auswahl liefert einen differenzierbaren Null-Loss. Ein gesamter
Train-/Val-Durchlauf ohne gültige maskierte Targets wird als Konfigurations- oder
Datenfehler abgewiesen. Nichtendliche Modelloutputs brechen den Trainer ab.
Zusätzliche Residual- oder Auxiliary-Aufgaben sind nicht implementiert.

CSV und TensorBoard loggen für `train/` und `val/`: `loss_total`, `loss_xyz`,
`loss_range`, `loss_intensity`, `masked_valid_ratio`, `valid_ratio`.
Die Ratios beziehen sich auf das aktuelle Frame; `masked_valid_ratio` ist der
maskierte Anteil an dessen gültigen Pixeln. Globale DDP-Mittel verwenden die
jeweiligen Komponentensummen und tatsächlich berücksichtigten Elementzahlen;
auch der Trainingsgradient wird entsprechend gewichtet. Validation wird ohne
Padding über Ranks verteilt. BatchNorm-Buffer werden davor synchronisiert.

## Checkpoints, Resume und Transfer

MAE speichert `checkpoints/last.pt` und `checkpoints/best_val_loss.pt`.
Formatkennung: `temporal_e2e_masked_reconstruction_v1`.
Enthalten sind voller Modell-State, separate Encoder-/Temporal-/Decoder-States,
AdamW-, Scheduler- und Scaler-State, Epoche, bestes Validation-Loss,
vollständige aufgelöste Konfiguration, Eingabekanäle/-reihenfolge,
Normalisierung, Projektions- und Modellparameter, Offsets und Scanrate.
Zusätzlich werden Python-, NumPy-, Torch-/CUDA- und Loader-Generator-Zustände
pro Rank gespeichert.

`--resume PATH_TO_LAST_PT` setzt am Epochenübergang fort. Für einen
reproduzierbaren Vergleich werden dieselben Modell-, Daten-, Augmentations-,
Masken- und Trainingskonfigurationen sowie dieselbe DDP-Weltgröße verlangt.
Verwende die gespeicherte `config.yaml`; CLI-Overrides dürfen diese Bedingungen
nicht verändern. Jeder Aufruf erstellt ein neues Logverzeichnis. Ein vorhandenes
`best_val_loss.pt` neben dem Resume-Checkpoint wird dorthin übernommen.
Die alte Datei bleibt erhalten. Einzelne Zwischen-Batches einer Epoche werden
nicht fortgesetzt; GPU-Kernel-Reproduzierbarkeit hängt weiterhin von der Umgebung ab.

Der MOS-Trainer lädt vor Optimizer-Erstellung und DDP-Wrapping sämtliche
`encoder.*`, `temporal.*` und `decoder.up_block1.*` bis `up_block4.*` inklusive
Buffer. Nur die beiden Head-Parameter werden ausgeschlossen. Format/Metadaten,
Kanäle, deren Reihenfolge, Normalisierung, Architektur, vollständige Schlüssel,
Shapes, Dtypes und endliche Gewichte werden strikt geprüft. Offset-Abweichungen
geben eine Warnung; sie verhindern den architektonisch möglichen Transfer nicht.
Der neue Zweikanal-Head bleibt frisch initialisiert, alle Parameter bleiben
trainierbar, und MOS erhält einen neuen AdamW ohne MAE-Optimizer-State.
MOS-Loss, Metriken, KNN, Threshold-Auswahl und Checkpoint-Auswahl bleiben erhalten.

## Startbefehle

Alle Befehle vom Repository-Verzeichnis, in der bisherigen Container-/HPC-Umgebung.
Die YAML-Datenpfade verwenden wie die Baseline `/hpc/SemanticKitti/...`.

P1, zwei P100-GPUs, 50 Epochen FP32-Pretraining:

```bash
bash submit_temporal_e2e_mae_p1_2gpu.sh
```

Der Wrapper verwendet die vorhandene P1-Container-/DDP-Infrastruktur von
`submit_mos.sh` und `train_mos.slurm`. Letzteres unterstützt jetzt einen auf die
beiden erlaubten Skripte begrenzten `TEMPORAL_TRAIN_ENTRY`; ohne diesen Wert
bleibt der bisherige MOS-Start unverändert. Batch 8 pro GPU ergibt global 16.
Die generischen Launch-Logs liegen weiterhin unter `/hpc/logs/mos/CONFIG/job-ID/`,
während die MAE-Checkpoints/CSV/TensorBoard dort in eindeutig benannten
`Temporal_E2E_MAE_...`-Runverzeichnissen liegen. Bei direktem Start verwendet
MAE die eigene Logwurzel `/hpc/logs/temporal_e2e_mae`.

Auf einer BF16-kompatiblen GPU direkt im Container:

```bash
python tools/train_temporal_e2e_mae.py \
  --cfg_path src/configs/semanticKitti_temporal_e2e_mae_intensity_off8_64x512_aug.yaml \
  --device cuda
```

Danach P1-MOS-Fine-Tuning mit zwei GPUs:

```bash
MAE_CHECKPOINT=/hpc/logs/mos/CONFIG/job-ID/Temporal_E2E_MAE_RUN/checkpoints/best_val_loss.pt
bash submit_mos.sh \
  src/configs/mos_semanticKitti_temporal_e2e_mae_intensity_off8_64x512_lovasz_w290_aug.yaml \
  --pretrained_path "$MAE_CHECKPOINT"
```

`MAE_CHECKPOINT` durch den vom Pretraining ausgegebenen **Container-Pfad** ersetzen.
Die Fine-Tuning-YAML behält 100 Epochen, CE+Lovász und sämtliche übrigen
MOS-Baseline-Bedingungen bei.

Kurzer P1-Smoke-Job (nur nach eigenem Start):

```bash
bash submit_temporal_e2e_mae_p1_2gpu.sh \
  src/configs/semanticKitti_temporal_e2e_mae_intensity_off8_64x512_aug_p1_2gpu_fp32.yaml \
  --epochs 1 --batch_size 1 --num_workers 0 --max_batches 1
```

Resume im Container:

```bash
python tools/train_temporal_e2e_mae.py \
  --cfg_path PATH_TO_PREVIOUS_RUN/config.yaml \
  --resume PATH_TO_PREVIOUS_RUN/checkpoints/last.pt --device cuda
```

## Tests und Grenzen

```bash
python -m unittest discover -s tests -p 'test_temporal_e2e_mae.py' -v
python tools/train_temporal_e2e_mae.py --device cpu --dry-run
python tools/test_temporal_mos_e2e.py
python -m unittest discover -s tests -p 'test_mos_*.py' -v
python -m unittest discover -s tests -p 'test_temporal_mos_augmentation.py' -v
python -m unittest discover -s tests -p 'test_temporal_mos_distributed.py' -v
```

Neue Tests prüfen T=2/3/4 und dynamische Offset-Listen, labelunabhängigen Loader,
identische MOS-/MAE-Eingaben, Maskierungsrate/Seeds/Sensorvalidität und Encoder-
Input ohne Target-Leak, exakte Loss-Werte, NaN-/Leermasken, Gradienten und echte
Updates in Encoder/Attention/UpBlocks/Head, vollständigen Transfer und dessen
Fehlerfälle, MOS-Trainingsschritt nach Transfer, Logging/Checkpoint-Integration,
exakten CPU-Resume-Vergleich, History-Diagnose sowie CPU-DDP mit Workern.
Auch der bestehende RangeViewSiamWCA-T-MAE-Pfad wird vorwärts/rückwärts geprüft.

`--history-diagnostic` vergleicht den normal maskierten Input mit fehlender
Historie und kopiertem **maskierten** aktuellen Frame. Gemessen werden mittlere
absolute Änderungen maskierter Rekonstruktionswerte und Loss, ohne eine
Verbesserung vorauszusetzen. Der optionale CLI-Lauf ist derzeit für einen
Prozess vorgesehen; die Funktion ist separat in Tests nutzbar. Bei einem frisch
initialisierten Netz können Änderungen sehr klein sein (Gamma startet bei 1e-3).

Lokal am 2026-10-28: PyTorch 2.14.1 CPU; CUDA nicht verfügbar, NVIDIA-Treiber
nicht erreichbar. Synthetischer 64×512-Forward/Backward mit allen Komponenten
war erfolgreich. Reale SemanticKITTI-Scans sind erreichbar: ein Trainings- und
Validation-Batch bei 64×512/Batch 1 mit Augmentation und Offset 8 liefen auf CPU
durch; Loss 13.888315 / 19.364648, Checkpoints mit ausschließlich endlichen
Modellzuständen gespeichert. Diese Werte sind reine Funktionsprüfungen.
Es wurde kein vollständiges Pretraining und kein Slurm-Job gestartet.
GPU-AMP und GPU-NCCL konnten hier nicht geprüft werden.
Die bereits dokumentierte Circular-Shift-/Azimuth-Binning-Einschränkung der
Baseline-Augmentation bleibt bestehen.

## Dateiliste und abschließende Verifikation

Neu:

- `src/helper/temporal_e2e_input.py`
- `src/helper/temporal_e2e_transfer.py`
- `src/temporal_e2e_mae_dataset.py`
- `src/loss_temporal_e2e_mae.py`
- `tools/train_temporal_e2e_mae.py`
- `tests/test_temporal_e2e_mae.py`
- die drei oben genannten MAE-/Fine-Tuning-YAMLs
- `submit_temporal_e2e_mae_p1_2gpu.sh`
- `TEMPORAL_E2E_MAE.md`

Geändert:

- `src/mos_models/salsanext_temporal_e2e.py`: zwei/fünf Ausgabekanäle,
  sonst dieselbe Feature-Verarbeitung und State-Dict-Schlüssel.
- `src/temporal_mos_e2e_dataset.py`: vorhandene Verarbeitung in gemeinsamen
  Mixin verschoben, öffentliche MOS-Rückgabe erhalten.
- `src/helper/mos_augmentation.py`: Rotation unterstützt labelunabhängige
  Eingaben mit `y=None`, ohne zusätzlichen Zufallszugriff.
- `tools/train_temporal_mos_e2e.py`: vollständiger optionaler Transfer,
  CLI-Checkpointpfad und explizite Zweikanalprüfung für MOS.
- `train_mos.slurm`: auswählbarer, auf MOS/MAE begrenzter Trainingseinstieg;
  MOS bleibt der Default.

Endstand der Tests: 12 neue MAE-Tests bestanden, 13 Augmentationsregressionen
bestanden, 9 vorhandene Temporal-E2E-Tests bestanden, 1 vorhandener Temporal-DDP-
Test bestanden, 47 vorhandene MOS-Tests davon 45 bestanden / 2 CUDA-Tests
übersprungen. Insgesamt 80 bestanden, 2 übersprungen. Shell-Syntax und P1-
Wrapper mit ersetztem `sbatch` erfolgreich geprüft; `git diff --check` sauber.
