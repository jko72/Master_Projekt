# LMNet-Angleichung, Profil `lmnet_v1`

Stand: 29.10.2026. Die Erweiterung wird mit der neuen vollständigen Config
`src/configs/mos_semanticKitti_temporal_e2e_lmnet_v1.yaml` gewählt.
Bestehende Configs, Checkpoints, Ergebnisse und Cacheordner wurden nicht ersetzt.
Ohne `mos_data_params.projection_mode` gilt weiterhin `legacy`.
MF-MOS ist nicht Teil dieser Implementierung.

Zusätzlich gibt es jetzt ein explizit wählbares, labelunabhängiges
LMNet-MAE-Pretraining. Die bisherigen MAE-Configs bleiben im Legacy-Pfad.
Config, p1-Start mit zwei GPUs/Globalbatch16, Maskenskalierung auf 8×64,
Checkpointtransfer und Tests sind in [LMNET_MAE_PRETRAINING.md](LMNET_MAE_PRETRAINING.md)
dokumentiert.

## Referenz und übernommene Regeln

Referenz ist das deterministische SalsaNext-MOS-Training von
[PRBonn/LiDAR-MOS am Commit 4f2e96f46f5ac267cf00e404ec94514f91ebfbf1](https://github.com/PRBonn/LiDAR-MOS/tree/4f2e96f46f5ac267cf00e404ec94514f91ebfbf1).
Die folgenden Dateien wurden am fixierten Commit gelesen:

| Quelldatei unter diesem Commit | Übernahme |
| --- | --- |
| [mos_SalsaNext/salsanext_mos.yml](https://github.com/PRBonn/LiDAR-MOS/blob/4f2e96f46f5ac267cf00e404ec94514f91ebfbf1/mos_SalsaNext/salsanext_mos.yml) | 64×2048, FoV +3/−25°, fünf μ/σ, Batch1, SGD .01/.9/.0001, Warmup1, Decay .99; KNN 5/5/1/1 |
| [dataset/kitti/parser.py](https://github.com/PRBonn/LiDAR-MOS/blob/4f2e96f46f5ac267cf00e404ec94514f91ebfbf1/mos_SalsaNext/train/tasks/semantic/dataset/kitti/parser.py) | Äußeres Augmentationsgate .5, innere Gates .5, q∼U(0,.5); Rohlabels und vorab berechnete Residuen |
| [common/laserscan.py](https://github.com/PRBonn/LiDAR-MOS/blob/4f2e96f46f5ac267cf00e404ec94514f91ebfbf1/mos_SalsaNext/train/common/laserscan.py) | yaw=−atan2(y,x), floor/Clamp, nächster Punkt gewinnt; Y-Spiegelung vor gemeinsamer Translation; Löschen mit Zurücklegen aus [0,N−2] |
| [modules/trainer.py](https://github.com/PRBonn/LiDAR-MOS/blob/4f2e96f46f5ac267cf00e404ec94514f91ebfbf1/mos_SalsaNext/train/tasks/semantic/modules/trainer.py) | Gewichte 1/(aggregierte Häufigkeit+.001), SGD, Scheduler nach Update, Auswahl nach Val-Mean-IoU |
| [common/warmupLR.py](https://github.com/PRBonn/LiDAR-MOS/blob/4f2e96f46f5ac267cf00e404ec94514f91ebfbf1/mos_SalsaNext/train/common/warmupLR.py) | LR-Verlauf gegen ausführbaren Originalcode getestet |
| [modules/Lovasz_Softmax.py](https://github.com/PRBonn/LiDAR-MOS/blob/4f2e96f46f5ac267cf00e404ec94514f91ebfbf1/mos_SalsaNext/train/tasks/semantic/modules/Lovasz_Softmax.py) | Vorhandene FP32-Implementierung weiterverwendet: present, per_image=false, λ=1 |
| [config/labels/semantic-kitti-mos.yaml](https://github.com/PRBonn/LiDAR-MOS/blob/4f2e96f46f5ac267cf00e404ec94514f91ebfbf1/mos_SalsaNext/train/tasks/semantic/config/labels/semantic-kitti-mos.yaml) | MOS-Mapping, Autorenhäufigkeiten und Splits |
| [utils/gen_residual_images.py](https://github.com/PRBonn/LiDAR-MOS/blob/4f2e96f46f5ac267cf00e404ec94514f91ebfbf1/utils/gen_residual_images.py), [utils/utils.py](https://github.com/PRBonn/LiDAR-MOS/blob/4f2e96f46f5ac267cf00e404ec94514f91ebfbf1/utils/utils.py), [config/data_preparing.yaml](https://github.com/PRBonn/LiDAR-MOS/blob/4f2e96f46f5ac267cf00e404ec94514f91ebfbf1/config/data_preparing.yaml) | Residualfilter 2<r<50 vor Kollisionen, Egomotion, abs(r_current−r_past)/r_current, Null am Sequenzanfang |

Projektions- und Scheduler-Testreferenzen sowie die Label-YAML liegen in
`tests/reference/lmnet_*`; die ursprüngliche MIT-Lizenz und Urheberangaben sind
unter `LICENSES/LiDAR-MOS-MIT.txt` erhalten. Die vorhandenen Lovász-Lizenzhinweise
bleiben erhalten.

## Config und Datenpfad

Das Vergleichsprofil nutzt weiterhin Historie `[8]`, `[x,y,z,range,intensity]`,
zwei Klassen und die vorhandene Temporalarchitektur mit bottleneck_only-Fusion.
Grundlage war die vorhandene Fullres-CE+Lovász-Config
`mos_semanticKitti_temporal_e2e_intensity_off8_fullres_lovasz.yaml`.

Es setzt 100 Epochen, Batch1, einen Prozess, Accumulation1, SGD mit LR=.01,
Momentum=.9, WeightDecay=.0001, Warmup1, Decay=.99 pro Epoche, AMP=false,
Clip=0 und pretrained_path=null. `checkpoint_metric: mean_iou` wählt
`best_mean_iou.pt`; `best_moving_iou.pt` wird zusätzlich gespeichert.
Andere Batch-/DDP-Varianten bleiben möglich; effektive Batchgröße und
Abweichungen stehen in `run_metadata`. Es gibt keine automatische LR-Skalierung.

`paper_normalization` standardisiert nach Messgrößennamen:
x 10.88/11.47, y .23/6.91, z −1.04/.86, range 12.12/12.32,
intensity .21/.16. Augmentation liegt davor, ungültige Pixel danach bei Null.
Die Benchmark-KNN erhält separat die unnormalisierte Reichweite in Metern.
Residualkanäle werden nicht mit diesen fünf Größen standardisiert.

CE-Gewichte: Static=1.0209899532210003, Moving=296.43707590294014;
Verhältnis=290.34279423391575. Die Tests aggregieren sie erneut aus der
fixierten Autoren-YAML. CE, Lovász und Gesamtverlust werden getrennt geloggt.

`mos_augmentation.mode` akzeptiert `none`, `lmnet`, `custom`.
Ohne Modus funktionieren die alten enabled-/Operationsschalter weiter.
LMNet erfordert die neue Projektion; widersprüchliche Custom-Schalter werden
abgelehnt. Im neuen Profil sind Rotation, historisches Bernoulli-Dropout und
Intensitäts-Jitter deaktiviert. Sie bleiben mit `custom` wählbar.
LMNet-Geometrie wird erst nach Egomotion auf alle Scans synchron angewandt;
nur aktuelle Punkte und ihre Labels werden gemeinsam gelöscht.

Im Training werden uint32-Rohlabels mit `&0xFFFF` und dem vorhandenen Mapping
online projiziert. Der Index benötigt dafür keine Bildlabelcaches und führt
keinen vollständigen Projektions-/Statistiklauf beim Start durch.
`frames_with_moving=None`/`label_stats_deferred=True` kennzeichnen die vertagte
Statistik. Validierung nutzt standardmäßig versionierte Caches; mit
`online_validation_labels: true` ist sie ebenfalls online möglich.
Die gemeinsame Frame-Baseline nutzt im LMNet-Modus Rohpunktlabels online und
lädt Residuen weiterhin unverändert aus Dateien; sie berechnet sie bei
Augmentation nicht neu. Für sie wurde kein zusätzlicher enger Trainingsvergleich
mit eigenem Scheduler angelegt. Ihre bisherigen Schedulerdefaults bleiben gültig;
die Optimizerfactory wird mit Temporal E2E geteilt.

Auf offiziellen Testsequenzen 11–21 erzeugt der Labelgenerator keine GT-Caches.
E2E-Inferenz kann `y=None` verarbeiten. Der Benchmark exportiert weiterhin
eine Vorhersage je ursprünglichem Rohpunkt als uint32 mit IDs 9/251.

Der neue Pfad behält Sequenzanfänge, nominales negatives dt und alle
verfügbaren Historien. Fehlende Historie erhält Nullfeatures und valid=false.
Training, Trainingsvalidierung und Benchmark verwenden diese Regel.
`samples_rank*.jsonl` protokolliert Sample-ID, Phase, Historienverfügbarkeit und
LR; CSVs enthalten Framezahlen und erste/letzte Update-LR. Benchmark-Coverage
protokolliert jedes Sample und die Anzahl seiner Originalpunkte.
Der Val08-Benchmark verarbeitet ohne `--max-frames` alle 4.071 Scans.
Die Rolle von Sequenz07 wird aus den tatsächlich gespeicherten Splits abgeleitet.

## Cache-Gültigkeit

| Cache | Neues Profil | Legacy |
| --- | --- | --- |
| mos_labels / mos_labels_2048 / andere bisherige Labels | Keine Wiederverwendung | Mit ihrer ursprünglichen Config gültig |
| mos_labels_2048_lmnet_v1 | Neue Projektion, 64×2048, finite/r>0 ohne obere Grenze; für Val08 erzeugen | Nur explizit verwenden |
| bisherige residual_images_* | Keine Wiederverwendung im LMNet-Modus | Bisherige Nutzung bleibt erhalten |
| residual_images_lmnet_v1_{offset} | Neue Projektion, 2<r<50 vor Kollisionen; nur Residual-Baseline | E2E lädt diese Dateien nicht |

`manifest.json` enthält Version, SHA256-Fingerprint, Projektion, H/W/FoV,
Filter, Mappingversion, Offset/Residualnormalisierung und SHA256 der lokalen
poses.txt/calib.txt samt Posekonvention. Fehlende/inkompatible Manifeste oder
Residualdateien erzeugen Fehler. Bestehende unversionierte Dateien werden
nicht nachträglich als neue Caches übernommen. Normalisierung und Loss ändern
allein den geometrischen Fingerprint nicht. Auch passende Offline-Labels
ersetzen keine Onlineprojektion augmentierter Trainingspunkte.

## Abweichungen

| Bereich | LMNet | neue Pipeline | Grund der Abweichung | methodisch/numerisch/technisch | Fundstelle |
| --- | --- | --- | --- | --- | --- |
| Laufzeit | 150 Epochen | 100 | Expliziter Nutzerwunsch; andere Optimierungsdauer | methodisch | salsanext_mos.yml; neue Vergleichsconfig |
| Head/Ignore | 3 Softmaxplätze, Ignore0 | 2 Logits; Static0, Moving1, Ignore−1 | Native MOS-Schnittstelle bleibt erhalten; anderer Softmaxnenner | methodisch/numerisch | Autoren-trainer.py; mos_lovasz_loss.py |
| Modellumfang | Aktueller Scan und vorkalkulierte Residuen | Bestehendes Temporalmodell und Rohhistorie [8] | Bestehende Architektur/Offsets erhalten; kein Wechsel auf acht Rohhistorien | methodisch | Autoren-parser.py; neue Config; temporal_mos_e2e_dataset.py |
| Temporalaugmentation | Geometrie des aktuellen Scans; Residuen bleiben vorkalkuliert | Gemeinsame Spiegelung/Translation aller ausgerichteten Rohscans, aktuelles Dropout | Zeitliche Geometriekonsistenz; andere temporale Eingabeverteilung | methodisch | parser.py/laserscan.py; mos_augmentation.py; temporal_e2e_input.py |
| Fehlende Historie | Nullresiduen am Anfang | Nullfeatures plus valid=false, nominales dt | Adaption für rohe Temporalfeatures; alle aktuellen Frames bleiben erhalten | methodisch | gen_residual_images.py; temporal_e2e_input.py; evaluate_mos_benchmark.py |
| Index0 | proj_idx>0 | proj_idx>=0 | Gültiger erster Punkt wird beibehalten | technisch | Autoren-laserscan.py do_range_projection; mos_projection.py |
| Lossnumerik | NLL auf Log-Softmaxausgabe und Autoren-Lovász | Stabile CE auf Logits, vorhandener FP32-Lovász, Sonderfälle korrigiert | Numerische Stabilität bei gleicher binärer Zielkonvention | numerisch | Autoren-trainer.py/Lovasz_Softmax.py; mos_lovasz_loss.py |
| Validierung | Parser verwendet drop_last=True | Vollständige Val08 ohne Weglassen von Randframes oder Batches | Alle 4.071 Scans und konsistente Punktmetrik | methodisch | Autoren-parser.py; Trainer make_loader; Benchmark |
| N=0/1 beim Dropout | randint(0,N−1,...) kann scheitern | Keine Löschung bei N<2 | Definierter technischer Sonderfall | technisch | Autoren-laserscan.py open_scan; mos_augmentation.py raw_scans |
| Ungültige Messungen | Keine entsprechende explizite Absicherung | Endliche Features, r>0; leere Pixel Null | Definierte Datenqualität, betrifft ungültige Messungen | technisch | mos_projection.py |
| RNG | Python random und NumPy | Torch-RNG bei denselben geforderten Verteilungen | Worker-/Resume-Konsistenz im vorhandenen Trainer; gleicher Seed garantiert keine identischen Autoren-Ziehungen | technisch | parser.py/laserscan.py; mos_augmentation.py |
| Warmupnumerik | Verschachtelter CyclicLR und warmupLR | Expliziter Updatezähler, gegen Original mit PyTorch2.14.1 getestet | Vollständig serialisierbarer Zustand; historische PyTorchversion kann Initialisierung anders handhaben | numerisch/technisch | warmupLR.py; mos_optimization.py; LR-Paritätstest |
| Batch/DDP-Varianten | YAML Batch1 | Vergleich Batch1/World1; andere Werte erlaubt und markiert | Bestehende HPC-Varianten erhalten; andere Gradienten/ggf. lokaler Lovász | methodisch | neue Config; Trainer run_metadata |
| CPU-/Workerwahl | Autorenworkers4 | Vergleich num_workers=0 | Lokal ausführbar und exakt getestetes Resume; bei anderen Workerzahlen kein identischer RNG-Verlauf versprochen | technisch | neue Config; Resume-Test |

Dies ist ein LMNet-naher Vergleich mit dokumentierten Temporaladaptionen,
keine vollständige LMNet-Reproduktion und kein Nachweis einer Leistungsverbesserung.

## Ausgeführte Prüfungen

Am 29.10.2026 mit PyTorch 2.14.1+cpu:

- 47 vorhandene MOS-Tests erfolgreich, zwei davon wegen fehlender CUDA übersprungen.
- 26 vorhandene Temporal-/MAE-Tests erfolgreich, einschließlich CPU-DDP/Worker.
- 13 neue LMNet-Tests: Autoren-Projektionsparität, Index0, Nah-/Fernbereich,
  Filter vor Residualkollisionen, Feature-/Target-/Punktzuordnung, gemeinsame
  Geometrie nach Egomotion, synchrones Löschen, Custom-Modus, Cache-Mismatch,
  Online-/Cache-/Benchmark-Parität, Testinferenz ohne GT und 9/251-Export.
- Gemeinsamer Abschlusslauf: 86 Tests, erfolgreich mit zwei CUDA-Skips;
  Log `/tmp/lmnet-final-regression.log`.
- LR-Parität gegen die fixierte Originalklasse für 1/4/10 Updates pro Epoche,
  einschließlich erster LR, Warmupende, Epochenübergang und Resume.
  Bei U Warmupupdates: erste Update-LR=.01/U, Peak bei Update U und U+1,
  danach .99**(1/U) pro erfolgreicher Aktualisierung. Schedulerstep liegt nach
  optimizer.step; bei AMP-Overflow wird der Scheduler nicht vorgerückt.
- Begrenzter Trainings-/Resume-Test: ununterbrochene zwei kleine Epochen
  gegen eine kleine Epoche plus Resume; Modellgewichte exakt gleich,
  Schedulerzustand und Optimizergruppen gleich. Resume unterstützt
  Epochengrenzen und lehnt geändertes Protokoll/Worldsize ab.
- Ein synthetischer Forward/Backward/SGD-Schritt mit Input [1,2,5,64,2048]
  bestand mit endlichem Loss=1.36656.
- Ein begrenzter echter SGD-Schritt auf Val08/000008 bei 64×2048 bestand:
  123.368 Rohpunkte, 97.782 gültige Pixel, CE=.7778458,
  Lovász=.6649426, Summe=1.4427884. Die Parameter änderten sich und die
  Dataset-/Benchmarkeingaben stimmten überein. Keine Leistungsbewertung.
- Die ersten neun Val08-Frames wurden als kleine Stichprobe indiziert,
  Frame0 hatte Nullhistorie; die lokale Val08-Rohdateizahl ist 4.071.

Logs der Ausführung: `/tmp/lmnet-mos-regression.log`,
`/tmp/lmnet-temporal-regression-unrestricted.log`, `/tmp/lmnet-alignment-tests.log`,
`/tmp/lmnet-fullres-smoke.log`, `/tmp/lmnet-real-smoke-result.json`.
Die anfänglichen Sandbox-DDP-/Workerfehler wurden durch erlaubte Ausführung
mit lokalen Sockets aufgelöst. Die Original-Warmup-Testklasse meldet beim
Konstruktor eine PyTorch-Warnung; der LR-Paritätsvergleich besteht trotzdem.

Offen bleiben GPU-Prüfungen (CPU-PyTorch; nvidia-smi erreicht keinen Treiber),
vollständige Cacheerzeugung, 100-Epochen-Training und vollständige Val08-
Punktauswertung. Es wurden keine vollständigen Trainings, Datenneuberechnungen
oder HPC-Jobs gestartet. Alte Checkpoints bleiben evaluierbar, erhalten aber
keine nachträglich erfundene Scheduler-/RNG-Resumehistorie.

## Direkt ausführbare Befehle in Reihenfolge

### P2 mit zwei GPUs und vollständig online projizierten Labels

Die zusätzliche vollständige Config
`src/configs/mos_semanticKitti_temporal_e2e_lmnet_v1_p2_2gpu_online.yaml`
aktiviert Online-Labels auch für die Validierung und deaktiviert die
Offline-Labelerzeugung. Keine Label- oder Residualcaches müssen vorbereitet
werden; originale Scans, Rohlabels, Posen und Kalibrierung bleiben notwendig.
Der bestehende Launcher bindet die Daten schreibgeschützt nach `/hpc` ein.
Die Auswahl `gpu:2` enthält keine GPU-Typbindung. Batch8 pro GPU ergibt
Globalbatch16 bei Worldsize2 und Accumulation1, eine protokollierte Abweichung vom Batch1/World1-
Vergleich. SGD-LR=.01, 100 Epochen und Historie [8] bleiben erhalten.

```bash
cd /home/koehlbrandt/hpc/LiDARSensorForecaster
P2_GPU_GRES=gpu:2 bash submit_mos_p2_2gpu.sh \
  src/configs/mos_semanticKitti_temporal_e2e_lmnet_v1_p2_2gpu_online.yaml
```

Der Befehl reicht den Job ein; er wurde vom Agenten nicht ausgeführt.
Geprüft wurden die Config mit dem tatsächlichen Configloader und die
Shellsyntax von Submit-/Slurmskript.

### Enger Vergleich mit Batch1/World1

Die folgenden Befehle sind für eine Bash im Repository. Pfade werden in einer
neuen Laufzeitconfig angepasst; vorhandene Configs bleiben unverändert.
Für GPU-Läufe einen passenden CUDA-Pythoninterpreter über PYTHON wählen.

```bash
cd /home/koehlbrandt/hpc/LiDARSensorForecaster
export PYTHON=/tmp/mos-lovasz-venv/bin/python
export DATASET_ROOT=/home/koehlbrandt/hpc/SemanticKitti/dataset/sequences
export LMNET_RUN_ROOT=/home/koehlbrandt/hpc/logs/mos/lmnet_v1
export LMNET_CFG=/tmp/mos_temporal_lmnet_v1.yaml
export LMNET_RESIDUAL_CFG=/tmp/mos_residual_prepare_lmnet_v1.yaml
export OMP_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/mos-mpl
"$PYTHON" - <<'PY'
import os, pathlib, yaml
for source, target in [
    ('src/configs/mos_semanticKitti_temporal_e2e_lmnet_v1.yaml', 'LMNET_CFG'),
    ('src/configs/semanticKitti_residual_lmnet_v1.yaml', 'LMNET_RESIDUAL_CFG')]:
    cfg = yaml.safe_load(pathlib.Path(source).read_text())
    cfg['dataset_path'] = os.environ['DATASET_ROOT']
    if 'mos_log_params' in cfg:
        cfg['mos_log_params']['log_root'] = os.environ['LMNET_RUN_ROOT']
    pathlib.Path(os.environ[target]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

# 1. Kleine Prüfungen; die volle Suite braucht lokale DDP-/Worker-Sockets.
"$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v
"$PYTHON" tools/train_temporal_mos_e2e.py --cfg_path "$LMNET_CFG" --device cpu --dry-run

# 2. Neue geometrische Val08-Labels; Training projiziert seine Labels online.
# Dieser vollständige Erzeugungslauf wurde vom Agenten NICHT gestartet.
"$PYTHON" tools/generate_mos_labels.py --cfg_path "$LMNET_CFG" --seq_id 08

# 3. Volltraining (100 Epochen); ebenfalls NICHT vom Agenten gestartet.
# Bei CUDA-Interpreter hier --device cuda ergänzen. Kein torchrun für World1.
"$PYTHON" tools/train_temporal_mos_e2e.py --cfg_path "$LMNET_CFG"

# 4. Bestes Mean-IoU-Checkpoint des jüngsten Vergleichslaufs wählen.
export CHECKPOINT="$("$PYTHON" - <<'PY'
import os, pathlib
paths = list(pathlib.Path(os.environ['LMNET_RUN_ROOT']).glob('*/checkpoints/best_mean_iou.pt'))
print(max(paths, key=lambda p: p.stat().st_mtime))
PY
)"
"$PYTHON" tools/evaluate_temporal_mos_e2e.py --checkpoint "$CHECKPOINT" --split val

# 5. Vollständige Val08-Punktmetrik; KNN-Hauptzahl und Direct zusätzlich.
"$PYTHON" tools/evaluate_mos_benchmark.py --checkpoint "$CHECKPOINT" \
  --dataset-root "$DATASET_ROOT" --sequences 08 --mode both --postprocess knn \
  --moving-threshold 0.5 --knn-k 5 --knn-search 5 --knn-sigma 1 --knn-cutoff 1 \
  --output-dir "$LMNET_RUN_ROOT/val08_knn"
"$PYTHON" tools/evaluate_mos_benchmark.py --checkpoint "$CHECKPOINT" \
  --dataset-root "$DATASET_ROOT" --sequences 08 --mode both --postprocess direct \
  --moving-threshold 0.5 --output-dir "$LMNET_RUN_ROOT/val08_direct"
```

Optionale Fortsetzung eines abgebrochenen Laufs: `--resume` erwartet `last.pt`
aus einem neuen Lauf mit gespeicherten Optimizer-/Scheduler-/RNG-Zuständen.
Mit derselben Laufzeitconfig und dem konkreten Checkpointpfad:

```bash
"$PYTHON" tools/train_temporal_mos_e2e.py --cfg_path "$LMNET_CFG" \
  --resume "${CHECKPOINT%/*}/last.pt"
```

Optional ausschließlich für eine Residual-Baseline mit Offset1; E2E benötigt
diese Dateien nicht. Eine begrenzte Vorschau kann `--max_frames 3` ergänzen:

```bash
"$PYTHON" tools/generate_residuals.py --cfg_path "$LMNET_RESIDUAL_CFG"
```

Optionaler labelunabhängiger Testexport (keine Punktmetrik ohne GT):

```bash
"$PYTHON" tools/evaluate_mos_benchmark.py --checkpoint "$CHECKPOINT" \
  --dataset-root "$DATASET_ROOT" --sequences 11 12 13 14 15 16 17 18 19 20 21 \
  --mode point --postprocess knn --moving-threshold 0.5 \
  --export-predictions --submission-zip --output-dir "$LMNET_RUN_ROOT/test_knn"
```
