# Temporal E2E MOS – Variante B: Residual Prediction

## Motivation und Abgrenzung

**Residual Prediction ist eine Auxiliary-Trainingsaufgabe. Während der MOS-Inferenz werden keine Residualbilder berechnet oder als Netzwerkeingabe verwendet.**

**Ein Residualbild ist eine zeitliche Range-Differenz und kein Ground-Truth-Bewegungslabel. Auch Occlusions, Projektionseffekte und Messrauschen können Residualwerte erzeugen.**

Ziel ist eine zusätzliche Supervision der zeitlichen Merkmale für den bereits
bestehenden Siamese-Encoder mit Local Historical Cross-Attention. Variante A
verwendet semantische Beweglichkeit (Weighted CE + Lovasz); Variante B verwendet
kontinuierliche zeitliche Range-Differenzen (Masked Smooth-L1). Beide sind
getrennte, gegenseitig ausschließende Betriebsarten desselben Trainers.

LMNet und MF-MOS verwenden Residualbilder als Netzwerkeingabe einer Motion
Branch. Hier lernt der gemeinsame Backbone, ein solches Bild vorherzusagen.
Es gibt weder eine neue Branch mit eigenem Encoder/Decoder noch direkte
Feature-Fusion, Residual-Gating oder zusätzliche Eingabekanäle. MAE-Pretraining,
TOP-Pretraining und dynamische Offsets wurden nicht hinzugefügt.

Geprüfte Ausgangsschnittstellen: `mos_auxiliary` aus Variante A,
`SalsaNextDecoder(return_features=True)`, `SalsaNextTemporalE2E(return_aux=True)`,
`TemporalMOSE2EDataset`, `TemporalE2EInputMixin`, `TemporalMOSAugmentation`,
`MOSCrossEntropyLovasz`, `run_phase`, strikter MAE-Transfer sowie beide
MOS-Evaluatoren. Der vorhandene Residual-Helper wurde tatsächlich geprüft.
Die bestehenden öffentlichen Residual-Funktionen bleiben unverändert.

## Architektur und Gradientfluss

```text
t−8 XYZRI ─┐
          ├─ Shared Siamese SalsaNext Encoder
  t XYZRI ─┘          ↓
             Local Historical Cross-Attention
                      ↓
             bestehende SalsaNext-UpBlocks
                      ↓
             gemeinsame 32 Decoderfeatures
                 ↙                  ↘
    decoder.logits (2)       residual_head (1)
     Static / Moving       relative Range-Differenz
```

`residual_head = Conv2d(32,1,1)` ist linear: kein Sigmoid, ReLU, Tanh oder
Clipping. Beide Heads erhalten exakt dasselbe Tensorobjekt; der Decoder wird
nur einmal ausgeführt. Der Residual-Loss führt Gradienten durch den neuen Head,
die Decoder-UpBlocks, die aktuellen Skipfeatures, die temporale Attention und
den Shared Encoder zurück. Es gibt kein `detach`.

Der isolierte Residual-Loss erzeugt keine Gradienten an `decoder.logits`.
Beim gemeinsamen Loss erhalten beide Heads Gradienten. Ein geänderter
Residual-Head verändert die MOS-Logits im selben eval-Forward nicht.
Die Modelle besitzen weiterhin fünf XYZRI-Eingabekanäle.

Parameter: MOS-only 7.075.906, Variante A 7.075.972, Variante B 7.075.939;
Variante B ergänzt genau 33 Parameter. Die Initialisierung des optionalen Heads
verwendet wie Variante A den isolierten CPU-RNG-Kontext mit Seed 1729.
Initiale gemeinsame Parameter und anschließende CPU-RNG-Folge bleiben identisch.

## Task-Auswahl und Konfiguration

MOS-only, auch weiterhin ohne Auxiliary-Block möglich:

```yaml
mos_auxiliary:
  enabled: false
```

Variante A behält ihre vollständige vorhandene Konfiguration und ihre Keys:

```yaml
mos_auxiliary:
  enabled: true
  task: movable
  loss_weight: 1.0
  movable:
    class_weights: [1.07047068699, 20.55648005566]
    loss:
      type: ce_lovasz
      lovasz_weight: 1.0
      lovasz_classes: present
      lovasz_per_image: false
    ignore_index: -1
  inference:
    use_auxiliary_output: false
```

Neue Datei: `src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_residual.yaml`:

```yaml
mos_auxiliary:
  enabled: true
  task: residual
  loss_weight: 0.3
  residual:
    offset: 8
    min_range: 2.0
    max_range: 50.0
    epsilon: 1.0e-6
    normalization: relative
    loss: smooth_l1
    beta: 0.1
    ignore_invalid_pairs: true
  inference:
    use_auxiliary_output: false
```

Die vollständige neue YAML übernimmt das tatsächliche LMNet-nahe
Baseline-Protokoll. Ein Test vergleicht alle Einträge gegen die unveränderte
Baseline-Protokollfixture und erlaubt ausschließlich neuen Run-Namen und
Auxiliary-Block als Unterschiede. Die bisherigen Movable- und MOS-only-YAMLs
wurden nicht geändert.

Insbesondere bleiben erhalten: 64×2048, FOV +3/−25°, `lmnet`/`lmnet_v1`,
Range 0/null für Inputs, XYZRI, `paper_normalization`, Offset[8], 10Hz,
Splits 00–07/09/10, 08, 11–21; Dropout .1, Attention-Heads8, Radius1,
Compare64, `bottleneck_only`; MOS CE+Lovasz, originale Klassengewichte;
SGD .01, Momentum .9, WeightDecay .0001, LMNet-Scheduler mit Warmup1/Decay.99;
100 Epochen, Seed 42, deterministic, AMPfalse, Batch8/GPU, zwei GPUs,
Accumulation1/global16, Clipping0, `mean_iou`, Scratch/null-Pretraining.
Auch Worker0, Logging- und übrige Baseline-Einstellungen sind übernommen.

Strikte Validierung erfolgt in Trainer, Dataset und Modell-Builder:

- Nur `movable` oder `residual`, keine gleichzeitigen Taskblöcke.
- Unbekannte Parameter werden auch in Unterblöcken verworfen.
- Lossgewicht endlich und ≥0; ohne Angabe taskbezogen 1.0 bzw. 0.3.
- Residualoffset positive ganze Zahl; `previous_offsets` muss genau diesen
  einen Offset enthalten. Kein stilles Umschalten auf einen anderen Scan.
- Residual unterstützt zunächst ausschließlich LMNet-Projektion.
- Rangegrenzen endlich und positiv, min < max; epsilon und beta endlich und >0.
- Ausschließlich `relative`, `smooth_l1`, `ignore_invalid_pairs:true`.
- `inference.use_auxiliary_output` muss false sein.
- Custom Historical Point Dropout wird bei aktiver Residual-Aufgabe abgelehnt,
  damit die Target-History sauber bleibt. Die neue YAML nutzt LMNet-Augmentation;
  bestehende MOS-/Movable-Custom-Augmentation bleibt unverändert verfügbar.

## Targetformel, Egomotion und Projektion

Die vorhandene Kalibrierungs-/Poseverarbeitung liefert Velodyne-Posen. Der
historische Scan wird mit

```text
T_past_to_current = solve(T_current, T_past)
                 = inverse(T_current) @ T_past
```

in das aktuelle LiDAR-System transformiert. Dafür wird die bestehende
`transform_points`-Funktion wiederverwendet. Der Offset bleibt8, also bei 10 Hz .8s.

Nach gemeinsamer geometrischer Augmentation werden beide Scans mit der
bestehenden `lmnet_projection` projiziert. Identisch zum MOS-Input bleiben
Floor/Clamp-Quantisierung, FoV-Konvention, Finite-Filter, Index0-Gültigkeit und
Kollisionsgewinner. Die Projektionsimplementierung wurde nicht verändert.
Die aktuellen und historischen Range-Bilder entstehen aus `Projection.ranges`
und `Projection.indices`, also **in Metern vor der Inputnormalisierung**.

```text
target(u,v) = abs(current_range(u,v) − aligned_history_range(u,v))
              / max(current_range(u,v), epsilon)
```

Keine log1p-Transformation, keine künstliche Begrenzung auf[0,1]. Gültige Werte
können größer1 sein; das ist getestet und in echten Daten beobachtet.
`target` ist float32[H,W], `valid_mask` bool[H,W]. Invalid-Targets werden als0
abgelegt, aber ausschließlich durch die separate Maske vom Loss ausgeschlossen.
Ein echtes Target0 bleibt gültig.

Gültigkeit setzt voraus:

```text
vorhandene History
AND belegter aktueller Clean-Pixel
AND belegter historischer Clean-Pixel
AND endliche Ranges
AND 2 < current_range < 50
AND 2 < aligned_history_range < 50
AND belegter aktueller Input-Pixel
AND gleiche ursprüngliche Current-Gewinnerpunkt-ID in Clean und Input
```

**Die Rangegrenzen gelten nur für die Auxiliary-Maske nach der vollständigen
Projektion.** Der MOS-Input bleibt im bisherigen unbegrenzten Rangebereich.
Das unterscheidet sich bewusst von `project_range_for_residual`, das in seiner
bisherigen LMNet-Referenz bereits Punkte vor der Kollision auf2–50m filtert.
Beispiel: Ein1m-Punkt vor einem10m-Punkt im selben Pixel bleibt Inputgewinner;
für Auxiliary wird der Pixel ungültig, statt den verdeckten10m-Punkt zu benutzen.
Die alten Residual-Funktionen und ihre bisherigen Werte wurden nicht verändert.
Neue Targets werden online berechnet; alte Residual-Caches werden nicht geladen.

## Geometrie, Point Dropping und Gewinneridentität

`raw_scans(..., return_details=True)` ergänzt optional:

```text
clean_scans: geometrisch augmentierte Scans vor Current-Punktlöschung
current_original_ids: ursprünglicher Index jedes überlebenden Current-Punkts
```

Ohne diese Option bleibt die bisherige Zweier-Rückgabe unverändert. Die
LMNet-Zufallsziehungen und deren Reihenfolge bleiben bestehen. Spiegelung und
Translation werden einmal gezogen und auf beide bereits ausgerichteten Scans
angewendet. Danach wird der aktuelle Input wie bisher gelöscht. Clean-Projektionen
verwenden dieselbe Geometrie vor dieser Löschung.

Der Input-Projektionsindex bezieht sich nach Löschung auf einen verkürzten
Punktvektor. Deshalb wird er zuerst über `current_original_ids` zurückübersetzt.
Nur bei Gleichheit mit dem ursprünglichen Clean-Gewinnerindex darf der Pixel
Residual-Supervision liefern. Gleiche Pixelkoordinaten allein reichen nicht.
Eine zufällig freigelegte Hintergrundbeobachtung wird ausgeschlossen.

Die Dataset-Tests belegen bitgleiche MOS-Inputs/Targets und identischen
anschließenden RNG-Zustand gegenüber MOS-only bei gleichem Seed. Variante A
behält ihr gemeinsames `[N,2]`-Labeldropping. Falls Range-View-Rotation aktiviert
ist, werden MOS, Residual und Residualmaske in einem Targetstack synchron gerollt.
Validation verwendet keine Zufallsaugmentation.

Translation kann die numerischen Ranges verändern. Das Target beschreibt daher
ausdrücklich die zeitliche Differenz **in der augmentierten gemeinsamen
Geometrie**, nicht zwingend den Wert aus den unveränderten Originalscans.
Targets, Masken und Punkt-IDs werden nur zur Supervision verwendet und nicht als
Eingabekanäle oder Feed-forward-Verbindungen in den MOS-Head eingespeist.

## Dataset, Modell und Inferenz

MOS-only unverändert:
`(x, valid, dt, y_mos, meta)`.

Movable unverändert:
`(x, valid, dt, y_mos, y_movable, meta)`.

Residual:

```python
(x, valid, dt, y_mos, aux, meta)
aux = {
    'task': 'residual',
    'target': residual_target,   # float32 [H,W]
    'valid_mask': residual_valid, # bool [H,W]
    'stats': {...},             # additive Projektions-/Filterzähler
}
```

Für diese Konfiguration: x[2,5,64,2048], valid[2,64,2048], dt[2],
y_mos[64,2048], Target und Maske[64,2048]. `collate_mos` stapelt das Dictionary
zu Batchdimensionen. Fehlende frühe History bleibt wie bisher Nullinput,
invalid und mit nominalem dt; sie liefert keine Residual-Supervision.

```python
mos = model(x, valid, dt)  # [B,2,H,W], nur MOS-Head berechnet
mos, residual = model(x, valid, dt, return_aux=True)  # residual [B,1,H,W]
(mos, residual), diagnostics = model(
    x, valid, dt, return_aux=True, return_diagnostics=True)
```

`return_aux=True` liefert bei Movable weiterhin[B,2,H,W]. Standard-Forward und
bisherige Diagnostik bleiben rückwärtskompatibel. Ein deaktiviertes Modell hat
keinen Auxiliary-Head.

Testsequenzen11–21 liefern `None` für MOS- und Auxiliary-Target. Der normale
Temporal-Input lädt wie bisher Current und vorhandene History; es werden weder
zusätzliche Scans noch Clean-Projektionen oder Residual-Ground-Truth berechnet.
Die regulären Evaluatoren bauen das Modell mit vollständiger Checkpointconfig,
laden strikt und rufen den Standard-Forward auf. Der Range-View-Evaluator fordert
wie schon bei Variante A ein MOS-only-Dataset an. Point-Benchmark, KNN,
Moving-Threshold und `.label`-Export wurden nicht verändert.

## Loss und DDP-Normalisierung

```text
L_MOS = unveränderte WeightedCE_MOS + Lovasz_MOS
L_Residual = Summe SmoothL1_beta(prediction, target) über gültige Pixel
             / Anzahl gültiger Pixel
L_Total = L_MOS + 0.3 * L_Residual
```

beta=.1, Aufgabenfaktor .3 ist eine neue experimentelle Einstellung und keine
MF-MOS-Autorengewichtung. Smooth-L1 arbeitet explizit in FP32 auch unter
Autocast. Ungültige Werte werden **vor** der Subtraktion ausgewählt; selbst NaN
in einem ignorierten Target trägt nichts bei. Bei0 gültigen Pixeln entsteht
eine differenzierbare Null. Kein CE/Lovasz auf dem Regressionskanal.

Bei Akkumulation über G Microbatches und R Ranks ist

```text
N_group = Summe aller gültigen Residualpixel über alle Ranks und Microbatches
Residual-Backward pro Rank und Microbatch = R * local_loss_sum / max(N_group,1)
```

Der Residualterm wird nicht zusätzlich durch G geteilt. DDP mittelt die
Rankgradienten; die Summe der akkumulierten Beiträge entspricht dadurch dem
globalen Pixelmittel der gesamten Gruppe. Auch die kürzere letzte Gruppe und
global leere Gruppen sind sicher.

Dazu werden die CPU-Batches einer Akkumulationsgruppe vorab geladen und ihre
Maskenzahlen einmal global summiert. Die Graphen werden weiterhin Microbatch
für Microbatch erzeugt und freigegeben. Zusätzlicher CPU-Speicher ist proportional
zur Akkumulationsgruppe; keine Speicherung aller GPU-Graphen. Bei Accumulation>1
kann das Vorladen die Interleaving-Reihenfolge des gemeinsamen CPU-RNG mit
Dropout ändern; die Augmentationsverteilung bleibt erhalten. Das Hauptprotokoll
hat weiterhin Accumulation1.

**MOS und Movable behalten ihre bisherige Normierung:** global gewichtete CE
pro Microbatch, ranklokale nichtadditive Lovasz, Mittel der Microbatches im
Akkumulationsfenster. Die neue Residual-Normierung ändert dieses bestehende
Lernziel nicht. Es gibt keine Kommunikation im Modell-Forward.

Validation verwendet das entpackte Modell nach Buffersynchronisation und
reduziert nur am Phasenende. Leere und unterschiedlich lange Shards führen zu
keinen schrittweisen Collectives. Optimizer, GradScaler, Clipping, Warmup,
Scheduler und Updatezählung werden im selben Trainer wiederverwendet.

## Logging, Quantile und Checkpoint-Auswahl

Bestehende CSV-Spalten bleiben bestehen und bedeuten weiter MOS:
`train_loss`, `train_loss_ce`, `train_loss_lovasz`, `train_loss_total`,
entsprechend für val sowie Moving-IoU/F1/Precision/Recall.

Zusätzlich je train/val:

- `aux_loss`, `aux_smooth_l1`, `aux_mae`, `aux_rmse`
- `aux_valid_ratio` (gültige Pixel / sämtliche Rasterpixel), `aux_valid_pixels`
- `aux_target_mean`, `aux_target_p50`, `aux_target_p90`, `aux_target_p99`
- `aux_target_near_zero_ratio` (Target ≤1e−6)
- `lambda_residual`, `aux_weighted_loss`, `aux_to_mos_loss_ratio`
- `total_loss` und `multitask_loss_total` (MOS + gewichteter Residualterm)
- `aux_current_pixels`, `aux_history_pixels`, `aux_clean_range_valid_pixels`,
  `aux_dropped_or_changed_winner_pixels`
- `parameter_count`, `phase_seconds` (maximale Rankzeit inkl. Datenladen),
  bestehende `peak_vram_bytes` und Computezeit pro Sample

Der bestehende TensorBoard-Schreiber erzeugt zusätzlich die entsprechenden
`train/...`- und `val/...`-Tags. Summen, Pixelzahlen, MAE-/MSE-Zähler und
Targetmittel werden global additiv aggregiert, nicht aus Rankmitteln gemittelt.

Für p50/p90/p99 wird ein mergebares 8192-Bin-Histogramm verwendet. Bei2–50m
liegt dessen Bereich bei[0,25], also Binbreite0.0030517578125. Die Werte werden
nicht geclippt; dies ist eine Statistikapproximation, keine Targettransformation.
Die lineare Quantilinterpolation erfolgt zwischen den beiden erforderlichen
Ordnungsstatistiken; deren Bin-Untergrenzen approximieren das exakte lineare
Samplequantil mit absolutem Fehler höchstens einer Binbreite.
`aux_target_quantile_bin_width` loggt die Auflösung. Die separate
Einzelscan-Diagnose berechnet zusätzlich exakte NumPy-Quantile.

Best-Checkpoint weiterhin ausschließlich MOS-`mean_iou`; gespeichert werden
`last.pt`, `best_mean_iou.pt`, `best_moving_iou.pt`. Residualmetriken sind kein
neues Auswahlkriterium.

## Checkpoint- und MAE-Kompatibilität

Alle bisherigen State-Keys bleiben bestehen, einschließlich
`decoder.logits.weight/bias` und `movable_head.weight/bias` für Variante A.
Variante B ergänzt ausschließlich `residual_head.weight/bias`.
Alle drei passenden Modell-/Checkpointkombinationen laden `strict=True`.
Eine Movable-/Residual-Verwechslung wird anhand von Keys und Shapes abgelehnt.

Gespeichert werden wie zuvor vollständige Modell-, Optimizer-, Scheduler-,
Scaler- und RNG-Zustände sowie vollständige Config und Auxiliary-Metadaten.
Resume vergleicht den gesamten Auxiliary-Block, die Task-/Loss-/Residualparameter,
Inputprojektion, Modellkonfiguration und Optimierung. Die feste Outputdimension
ist aus der Task definiert und wird beim strikten State-Laden überprüft.

MAE→MOS prüft weiterhin Format, Protokollmetadaten, erlaubte Schlüsselmenge,
Shapes, Dtypes und endliche Werte. Nur MOS-Head und jeweils aktiver Auxiliary-Head
bleiben frisch; Encoder, Attention und Decoder werden übernommen. Keine
pauschale Nutzung von `strict=False`, keine Änderung am MAE-Pretraining selbst.

## Startbefehle und Slurm-Logs

**Für neue Jobs den Submit-Wrapper verwenden:**

```bash
cd /home/koehlbrandt/hpc/LiDARSensorForecaster
bash submit_mos_p2_2gpu.sh \
  src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_residual.yaml
```

Er fordert weiterhin ohne Typbindung zwei GPUs auf einem P2-Knoten an; somit
ist keine40GB-Beschränkung eingeführt. Batch8 pro GPU/global16,100 Epochen.
Der Wrapper erstellt vor `sbatch` das Logverzeichnis und setzt Jobname sowie
absolute Output-/Errorpfade. Daraus entstehen:

```text
/home/koehlbrandt/hpc/logs/slurm/
  mos_semanticKitti_temporal_e2e_lmnet_aux_residual-<JOB-ID>.out
  mos_semanticKitti_temporal_e2e_lmnet_aux_residual-<JOB-ID>.err
```

Für Variante A denselben Wrapper mit der vorhandenen Movable-YAML verwenden.
Die `#SBATCH`-Fallbacks im Trainingsskript zeigen nun ebenfalls nach
`../logs/slurm/%x-%j.out/.err` statt ins Repository. Beim direkten `sbatch`
aus dem Repository muss dieses Verzeichnis bereits existieren; der generische
Jobname bleibt ohne Wrapper `mos-temporal`. Der Wrapper stellt den vollständigen
Konfigurationsnamen sicher. Slurm öffnet Logs vor Ausführung des Skripts;
deshalb werden diese Pfade beim Submit gesetzt. Bereits laufende Jobs und
bestehende Logdateien werden nicht verschoben oder verändert.

Ein kurzer GPU-Smoke im vorhandenen HPC-Container:

```bash
python tools/train_temporal_mos_e2e.py \
  --cfg_path src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_residual.yaml \
  --device cuda --epochs 1 --batch_size 1 --num_workers 0 --max_batches 10 \
  --log_root /hpc/logs/mos/residual_aux_smoke
```

Mindestens10 Validierungsbatches bei Batch1 sind sinnvoll, um über die ersten8
Frames ohne History hinauszukommen. Für reine Forward-/Backward-Prüfung:

```bash
python tools/train_temporal_mos_e2e.py \
  --cfg_path src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_residual.yaml \
  --device cuda --dry-run
```

Nach100 Epochen, für alle 4.071 Originalscans von08:

```bash
CKPT=/hpc/logs/mos/NEW_RUN/checkpoints/best_mean_iou.pt
python tools/evaluate_mos_benchmark.py \
  --checkpoint "$CKPT" \
  --dataset-root /hpc/SemanticKitti/dataset/sequences \
  --sequences 08 --mode point --postprocess knn \
  --knn-k 5 --knn-search 5 --knn-sigma 1.0 --knn-cutoff 1.0 \
  --moving-threshold 0.5 --device cuda \
  --output-dir /hpc/logs/mos/NEW_RUN/point08_knn5_residual_aux
```

Keine Änderungen an Point-Mapping, Poseverarbeitung, KNN oder Threshold.
Es wurde kein vollständiger HPC-Job gestartet.

## Ausgeführte Tests und reale Smokes

Neue Suite `tests/test_temporal_mos_aux_residual.py` mit 12 Tests:
Geometrie/Posen, Rangegrenzen und Finite-Filter, echte Nulltargets,
Kollisionsgewinner/Original-IDs nach Dropping, gemeinsame Geometrie und RNG,
Dataset-Shapes/Masken/MOS-Parität, fehlende History/Test ohne GT,
FP32-Smooth-L1 und ignorierte NaNs, isolierte und gemeinsame Gradienten,
Feature-Tensoridentität, MOS-only-Forward, strikte Checkpoints/Resume,
MAE-Transfer, Histogrammstatistik, echte Wrapper-Argumente mit simuliertem
`sbatch` sowie CPU/Gloo-DDP und optional CUDA.

Der DDP-Test vergleicht vollständige SGD-Parameterupdates gegen eine globale
Referenz mit **unterschiedlichen Pixelzahlen zwischen Ranks und Microbatches**.
Zusätzlich: ein lokal leerer Rank, eine global leere letzte Gruppe, Accumulation2,
verkürzte letzte Gruppe, globale Fehlerstatistiken, leere Validierungsshards und
zwei Backwards des echten Temporal-Modells ohne fehlende Parametergradienten.

Exakte lokale Befehle (Projektwurzel):

```bash
MPLCONFIGDIR=/tmp/mos-mpl OMP_NUM_THREADS=2 /tmp/mos-lovasz-venv/bin/python \
  -m unittest discover -s tests -p 'test_temporal_mos_aux_residual.py' -v
MPLCONFIGDIR=/tmp/mos-mpl OMP_NUM_THREADS=2 /tmp/mos-lovasz-venv/bin/python \
  -m unittest discover -s tests -p 'test_temporal_mos_aux_movable.py' -v
MPLCONFIGDIR=/tmp/mos-mpl OMP_NUM_THREADS=2 /tmp/mos-lovasz-venv/bin/python \
  -m unittest discover -s tests -p 'test_mos_*.py' -v
MPLCONFIGDIR=/tmp/mos-mpl OMP_NUM_THREADS=2 /tmp/mos-lovasz-venv/bin/python \
  -m unittest discover -s tests -p 'test_lmnet*.py' -v
MPLCONFIGDIR=/tmp/mos-mpl OMP_NUM_THREADS=2 /tmp/mos-lovasz-venv/bin/python \
  -m unittest discover -s tests -p 'test_temporal*.py' -v
bash -n submit_mos_p2_2gpu.sh train_mos_p2_2gpu.slurm
```

DDP wurde mit Freigabe außerhalb der Sandbox ausgeführt, da lokale
Gloo-Verbindungen darin blockiert waren. Stand 31.10.2026, PyTorch 2.14.1+cpu:

| Testsuite | Ergebnis |
|---|---|
| Residual | 12 Tests, OK,1 CUDA-Skip |
| Movable | 9 bestanden |
| MOS | 47 Tests, OK,2 CUDA-Skips |
| LMNet/MAE | 20 bestanden |
| Temporal inkl. beider Auxiliary-Suiten | 47 Tests, OK,1 CUDA-Skip |

MOS + LMNet + Temporal ergeben114 unterschiedliche Tests:111 bestanden,
3 mangels CUDA übersprungen. Die separat ausgeführten Auxiliary-Tests sind in
Temporal bereits enthalten und werden dabei nicht doppelt gezählt.

CPU-Synthetic mit der vollständigen neuen YAML und unveränderten64×2048:

```bash
MPLCONFIGDIR=/tmp/mos-mpl OMP_NUM_THREADS=2 /tmp/mos-lovasz-venv/bin/python \
  tools/train_temporal_mos_e2e.py \
  --cfg_path src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_residual.yaml \
  --device cpu --dry-run
```

Bestanden: Input[1,2,5,64,2048], MOS[1,2,64,2048], beide Heads,
Backward/Optimizer-Schritt, vorhandene/endliche Gradienten, Loss1.49241.

Realer SemanticKITTI-Smoke: temporäre Config unter `/tmp/residual-aux-real.yaml`,
ausschließlich lokaler Daten-/Logpfad und deaktiviertes TensorBoard;
CLI-Limits1 Epoche/Batch1/max_batches1. Vollauflösung und Offset 8 erhalten.
19.130 Trainings- und4.071 Validierungsframes indexiert, alle drei Checkpoints
gespeichert. Reproduzierbar:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path('src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_residual.yaml').read_text())
cfg['dataset_path'] = str(Path('../SemanticKitti/dataset/sequences').resolve())
cfg['mos_log_params'].update(log_root='/tmp/residual-aux-real-runs', use_tensorboard=False)
Path('/tmp/residual-aux-real.yaml').write_text(yaml.safe_dump(cfg))
PY
MPLCONFIGDIR=/tmp/mos-mpl OMP_NUM_THREADS=2 /tmp/mos-lovasz-venv/bin/python \
  tools/train_temporal_mos_e2e.py --cfg_path /tmp/residual-aux-real.yaml \
  --device cpu --epochs 1 --batch_size 1 --num_workers 0 --max_batches 1
```

Gemessener Trainingsbatch: MOS-Loss1.260188, Smooth-L1 .400934,
Gesamtloss1.380468;24.629 Clean-Paare nach Rangefilter, davon2.402 wegen
Punktlöschung/geändertem Gewinner ausgeschlossen,22.227 gültige Residualpixel
(16,96% des Rasters). MAE .448461, RMSE .578303. Dies ist ein Smoke nach einem
Update, kein wissenschaftliches Ergebnis.

Der erste Validation-Scan hat erwartungsgemäß keine t−8-History und somit0
Residualpixel, aber einen endlichen MOS-Loss1.202549. Zusätzlich wurde der
neugeladene echte Checkpoint auf einem **gültigen Paar von08/Frame 100** im
normalen Trainer-Validierungspfad ausgewertet:34.898 Residualpixel,
Smooth-L1 .122568, MAE .170951, RMSE .313068,
MOS-Loss1.207600 und Gesamtloss1.244370.

Der reale Checkpoint wurde anschließend im regulären Range-View-MOS-Evaluator
und im **unveränderten Point-KNN-Evaluator** mit einem Originalscan aus08
strikt geladen und erfolgreich ausgewertet (KNN5/5/1/1, Threshold .5).
Nach nur einem Trainingsschritt war Point-Moving-IoU0; daraus folgt keine
Aussage über die experimentelle Wirksamkeit.

Lokale Artefakte: `/tmp/residual-aux-real-runs/`,
`/tmp/residual-aux-point-smoke/`, `/tmp/residual-aux-valid-pair.json`.
Logs: `/tmp/residual-aux-tests.log`, `/tmp/residual-movable-regression.log`,
`/tmp/residual-mos-regression.log`, `/tmp/residual-lmnet-regression.log`,
`/tmp/residual-temporal-regression.log`, `/tmp/residual-aux-dry-run.log`,
`/tmp/residual-aux-real.log`, `/tmp/residual-aux-real-eval.log`,
`/tmp/residual-aux-point.log`, `/tmp/residual-aux-valid-pair.log`.

## Reale Targetdiagnose und gespeicherte Arrays

Neue separate Diagnosefunktion, ohne Änderung des MOS-Evaluators:

```bash
MPLCONFIGDIR=/tmp/mos-mpl OMP_NUM_THREADS=2 /tmp/mos-lovasz-venv/bin/python \
  tools/diagnose_temporal_mos_aux_residual.py \
  --dataset-root /home/koehlbrandt/hpc/SemanticKitti/dataset/sequences \
  --sequence 08 --frame 100 --output /tmp/residual-aux-scan08-frame100.npz
```

Dieser Befehl wurde tatsächlich ausgeführt. Er speichert metrisches aktuelles
Rangebild, ego-kompensiertes historisches Rangebild, tatsächliches
Current-Input-Rangebild, Target, Maske und originale Gewinnerindizes. Die
Ausgabedatei wird exklusiv neu angelegt; existierende Diagnoseartefakte werden
nicht überschrieben. Sie kann mit `numpy.load` geöffnet werden.

Für08/Frame 100:97.934 aktuelle und45.739 historische belegte Pixel;
34.898 gültige Paare nach2–50m-Filter,26,625% des Rasters bzw.35,63% der
aktuellen belegten Pixel. Keine stochastische Löschung in Validation.
Targetmittel .095688; exakte NumPy-Quantile:
p50 .0232012, p90 .1602672, p99 1.5010451.
AnteilTarget≤1e−6:0,002865%. Die Überlappung ist aufgrund von Projektion,
Sichtbarkeit und.8s Abstand kleiner als die aktuelle Scanbelegung; die
Rangegrenzen wurden nicht zur Erhöhung dieser Quote verändert.

## Einschränkungen und wissenschaftlicher Vergleich

| Experiment | Auxiliary | Loss/Faktor | Point-Moving-IoU08 |
|---|---|---|---|
| B0 | keine | MOS CE+Lovasz | bereitgestellte Baseline59,2235% |
| B1 | Movable | WCE+Lovasz,1.0 | noch zu bestimmen |
| B2 | Residual | Masked Smooth-L1,.3 | noch zu bestimmen |

Alle verwenden dieselben XYZRI-Eingaben, Offset[8], gemeinsame Hauptarchitektur,
MOS-Loss/Optimierung, Augmentationsverteilung, Checkpoint-Auswahl und Seedstrategie
sowie Originalpunkt-KNN-Evaluation. Keine Lambda-Ablationen wurden gestartet.

Noch erforderlich: GPU-Smoke inklusive AMP, zwei echte CUDA-/NCCL-Ranks,
VRAM bei Batch8/GPU, vollständiges100-Epochen-Training und Point-Auswertung über
alle 4.071 Scans. `nvidia-smi` hatte hier keine Treiberverbindung; CUDA war in
PyTorch nicht verfügbar. Die drei CUDA-Tests wurden ausdrücklich übersprungen,
nicht als bestanden gewertet. CPU-Peak-VRAM0 ist keine GPU-Speichermessung.
Es wurde kein externer HPC-Job eingereicht.

Residuals sind durch Occlusions, Range-View-Kollisionen, Messrauschen und
Posefehler beeinflusst. Der Zielraum ist eine dichte Pixelüberlappung und keine
Punktkorrespondenz oder Ground-Truth-Bewegung. Die Current-Gewinnerprüfung
verhindert die zusätzliche künstliche Supervision durch Input-Point-Dropping,
aber beseitigt diese grundsätzlichen Beobachtungseffekte nicht.
Die ersten8 Frames ohne History trainieren weiterhin MOS, nur ohne Residualterm.
Ein einzelner Seed erlaubt keinen belastbaren statistischen Wirksamkeitsnachweis.

## Dateiübersicht für Variante B

Neu: `TEMPORAL_MOS_AUX_RESIDUAL.md`,
`src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_residual.yaml`,
`src/helper/mos_residual_auxiliary.py`,
`tests/test_temporal_mos_aux_residual.py`,
`tools/diagnose_temporal_mos_aux_residual.py`.

Erweitert: `src/helper/mos_auxiliary.py`, `src/helper/residual_helper.py`,
`src/helper/mos_augmentation.py`, `src/helper/temporal_e2e_input.py`,
`src/helper/temporal_e2e_transfer.py`, `src/temporal_mos_e2e_dataset.py`,
`src/mos_models/builder.py`, `src/mos_models/salsanext_temporal_e2e.py`,
`tools/train_temporal_mos_e2e.py`.

Zusätzlich: `tests/test_temporal_mos_aux_movable.py` verwendet als ungültige Task
nun `unknown`, da `residual` jetzt gültig ist; seine bisherigen Prüfungen bleiben
erhalten. `train_mos_p2_2gpu.slurm` und `submit_mos_p2_2gpu.sh` dokumentieren und
sichern die gewünschte Logablage; die Startanleitung in
`TEMPORAL_MOS_AUX_MOVABLE_MFMOS.md` nutzt ebenfalls den Submit-Wrapper.
Die lokal ausgeschlossenen neuen YAML-/Launcherdateien wurden explizit für die
Git-Diff-Ansicht aufgenommen. Bestehende Trainings-/Ergebnisdateien bleiben
unangetastet.
