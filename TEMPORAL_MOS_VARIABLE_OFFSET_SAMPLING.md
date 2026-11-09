# Variable Temporal Offset Augmentation

## Experiment und Motivation

Das Experiment untersucht, ob variable historische Zeitabstände die Point-Moving-IoU
der bestehenden Siamese-/Local-Cross-Attention-Architektur gegenüber dem festen
Offset 8 verbessern. Referenz ist das cb_gate_mfmos-Modell mit ungefähr 61,14 % auf
SemanticKITTI Validation 08 (bisheriger Bestcheckpoint: Epoche 85).
Ein Ergebnis des neuen vollständigen Trainings liegt noch nicht vor.

Verbindliche, unveränderte Basis:
`src/configs/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate.yaml`.
Die Datei wurde aus dem tatsächlich eingebundenen HPC-Verzeichnis gelesen.
Auf dem Host liegt das Projekt unter `/home_cu/s210891/LiDARSensorForecaster`,
hier ist es unter `/home/koehlbrandt/hpc/LiDARSensorForecaster` eingebunden.
Das vorhandene Slurm-Skript bindet den HPC-Home-Ordner im Container auf `/hpc`.
Es wurde auf Branch `hpc/p1-ddp-mos`, Ausgangscommit `19e142d`, gearbeitet.

Neue vollständige Config:
`src/configs/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate_temporal_sampling.yaml`.
Gegenüber der Basis unterscheiden sich ausschließlich die Sampling-Sektion und
der Run-Name `MOS_temporal_e2e_lmnet_cb_gate_temporal_sampling_5_8_10_w1`.
Ein Test vergleicht die gesamten YAML-Inhalte auf diese Eigenschaft.
Die lokale `.git/info/exclude` schließt neue HPC-Experimentconfigs aus Git aus;
die neue Config bleibt entsprechend dieser bestehenden Regel lokal auf dem HPC.

## Ein historisches Frame, variable Zeitabstände

`previous_offsets: [8]` bleibt unverändert. Das Training wählt pro Zugriff auf
ein Sample genau einen historischen Offset. Es lädt niemals alle drei Historien
zusammen. Die Eingabe bleibt `[historisch, aktuell]`, also `x[B,2,5,H,W]`,
`valid[B,2,H,W]`, `dt[B,2]`. MOS- und Movable-Logits bleiben `[B,2,H,W]`.

| Offset | Wahrscheinlichkeit bei vollständig verfügbarer Historie | dt in Sekunden |
|---|---|---|
| 5 | 0,25 | `[-0.5, 0.0]` |
| 8 | 0,50 | `[-0.8, 0.0]` |
| 10 | 0,25 | `[-1.0, 0.0]` |

## Auswahl, Sequenzanfänge und fehlende Dateien

`TemporalMOSE2EDataset.__getitem__` in `src/temporal_mos_e2e_dataset.py`
ruft `_sample_previous_offset` nur bei `split == 'train'` und aktiviertem Sampling
auf. Für jeden Kandidaten werden Sequenzindex, tatsächliche Scan-Datei und eine
verfügbare, endliche 4×4-Pose geprüft. Die Auswahl erfolgt ausschließlich in der
Sequenz des aktuellen Samples; `self.offsets` wird niemals verändert.

Bei t=3 bleibt die Historie leer, mit Referenzoffset 8 und `history_available=[False]`.
Bei t=6 wird Offset 5 gewählt. Bei t=9 werden 5 und 8 mit Wahrscheinlichkeiten
1/3 und 2/3 gewählt. Ab t=10 gilt bei vollständig vorhandenen Scans/Posen die
Verteilung 0,25/0,50/0,25. Fehlende Dateien oder Posen können auch später die
Kandidatenmenge verkleinern; die Gewichte werden entsprechend neu normiert.
Kein aktuelles Trainingsframe wird wegen des Samplings entfernt.

Wenn kein Kandidat verfügbar ist, erzwingt der lokale `empty_history`-Parameter
die leere Historie. Dies gilt auch für andere zulässige Configs, deren
Referenzoffset nicht in der Kandidatenliste steht. Ein dann eventuell vorhandener
Referenzscan wird nicht als gezogene Historie ausgegeben.

Sonderfall bei anderen Configs: Sind ausschließlich Kandidaten mit Gewicht null
verfügbar, ist eine Normierung mathematisch undefiniert. Dann wird gleichverteilt
aus diesen vorhandenen Historien gezogen. Die Experimentconfig hat ausschließlich
positive Gewichte und erreicht diesen Sonderfall nicht.

## Scan, Egomotion, Projektion und Augmentation

Die bestehenden Funktionen in `src/helper/temporal_e2e_input.py` unterstützen
lokale optionale `offsets`, beispielsweise `(5,)`:
`_load_lmnet_temporal_input`, `_project_lmnet_temporal`, `_finish_lmnet_temporal`.
Bei `offsets=None` wird weiterhin `self.offsets` verwendet.
Der geladene Scan ist tatsächlich `t-offset`, und die Transformation bleibt
`solve(current_pose, selected_history_pose)`. Anschließend laufen die bisherige
LMNet-Projektion, gemeinsame Raw-Scan-Augmentation, Range-View-Augmentation und
Normalisierung. Der aktuelle Scan und seine MOS-/Movable-Labels stammen weiterhin
vom aktuellen Sample. `src/helper/mos_augmentation.py` wurde nicht geändert.

`dt` wird aus genau diesen lokalen Offsets als `-offset / scan_rate_hz` berechnet.
Das unveränderte Modell in `src/mos_models/salsanext_temporal_e2e.py` übergibt die
historische Spalte als `history_dt` an die unveränderte `LocalHistoricalAttention`
in `src/models/range_view_local_temporal_attention.py`.
Encoder, Decoder, Semantic Branch, Gate einschließlich Faktor 0,1, beide Losses,
Optimizer, Scheduler und Modellparameter wurden nicht verändert.

Das Dataset-Tuple bleibt mit Movable-Auxiliary:
`x, valid, dt, y_mos, y_movable, meta`. Ohne Auxiliary bleibt das bestehende
fünfteilige Tuple erhalten. Die Metadaten enthalten bei einer Historie
`sampled_previous_offset`, die tatsächlich verwendeten `previous_offsets`,
`history_paths` und `history_available`. Leere Historie wird mit leerem Pfad,
Nullfeatures und ungültiger Maske gekennzeichnet.

## Validation und Inferenz

Validation und Test benutzen ohne Zufallsziehung weiterhin `previous_offsets: [8]`.
`tools/evaluate_mos_benchmark.py` wurde nicht verändert: Sein eigener Input-Builder
liest die festen `previous_offsets` aus der Checkpointconfig und ignoriert die
Trainingssampling-Sektion. Tests vergleichen seine Eingaben und RNG-Zustände
mit und ohne neue Sektion.

Der zusätzliche bestehende Evaluator `tools/evaluate_temporal_mos_e2e.py` schaltet
Sampling in seiner lokalen Config-Kopie ebenfalls aus, auch bei `--split train`.
Es wurde kein neuer Evaluationsmodus implementiert.

## Config und Validierung

```yaml
mos_data_params:
  previous_offsets: [8]
  temporal_offset_sampling:
    enabled: true
    offsets: [5, 8, 10]
    probabilities: [0.25, 0.50, 0.25]
    train_only: true
```

`src/helper/temporal_offset_sampling.py` validiert zentral für Dataset und
`tools/train_temporal_mos_e2e.py`: Mapping, bekannte Schlüssel, Boolean `enabled`,
`train_only=true`, positive eindeutige Integer-Offsets ohne Boolean-Werte,
endliche nichtnegative Gewichte, gleiche Listenlängen, positive Gesamtmasse und
Summe 1 mit absoluter Toleranz 1e-6. Aktiviertes Sampling verlangt einen einzelnen
positiven Referenzoffset sowie LMNet und `salsanext_temporal_e2e`.
`mos_auxiliary.task=residual` wird ausdrücklich abgelehnt; Movable ist unterstützt.
Ohne die neue Sektion oder bei `enabled:false` bleibt Sampling aus.
Die Validierung schreibt keine neuen Defaults in die Config.

## RNG, DDP, Worker und Resume

Der Draw erfolgt mit `torch.multinomial` auf CPU. Es wird keine NumPy-RNG oder
neue Initialisierung pro Sample eingeführt. Ohne Sampling gibt es keinen
zusätzlichen Zufallsdraw. Das vorhandene Seeding bleibt `seed + rank` und der
DataLoader-Generator bleibt separat mit `seed + rank` initialisiert. PyTorch
initialisiert die Worker-RNGs; es gibt keinen global veränderlichen Offsetzustand.
`DistributedSampler.set_epoch(epoch)` bleibt erhalten.

Für identischen Seed, gleiche Umgebung, Workerzahl und Samplereihenfolge sind
die Offsetfolgen reproduzierbar. Verschiedene Seeds können unterschiedliche
Folgen liefern. Der bestehende Checkpoint speichert weiterhin Torch-/CUDA-/Python-/
NumPy-RNG sowie den Trainingsloader-Generator je Rank. Es werden keine zusätzlichen
Workerzustände oder Prefetch-Queues gespeichert; eine allgemeine bitweise
Reproduzierbarkeit für Multiworker-Resume oder Änderungen der Umgebung wird
nicht zugesichert. Es gibt keine neue Resume-Implementierung.

Die bestehende Resume-Validierung vergleicht `mos_data_params` vollständig und
lehnt somit feste gegen variable oder andere Sampling-Einstellungen ab.
Passendes Resume, Config-Speicherung und alte Configs/Checkpoints ohne Sampling
werden in einem echten Trainings-/Checkpoint-Roundtrip getestet.

## Training, Checkpoints und Statistik

Weiterhin `tools/train_temporal_mos_e2e.py`, von Scratch, 100 Epochen, Seed 42,
Batchgröße 8 pro GPU, zwei Worker pro Rank und alle sonstigen Basiseinstellungen.
Die Checkpointauswahl bleibt entsprechend der Basis `mean_iou`; der separate
`best_moving_iou.pt` wird wie bisher zusätzlich gespeichert und dient dem
gewünschten späteren Benchmarkvergleich.
Die vollständige Sampling-Sektion steht in `config_source.yaml`, `config.yaml`
und der im Checkpoint gespeicherten Config.

Pro Trainingsphase wird eine neue `TemporalOffsetStatistics` angelegt. Sie zählt
die tatsächlich verarbeiteten Batch-Metadaten im Hauptprozess. Am Ende gibt es
eine einzige zusätzliche DDP-Summenreduktion, keine Synchronisation pro Sample.
`epochs.csv` erhält:

- `train_offset_5_count`, `train_offset_8_count`, `train_offset_10_count`
- `train_offset_5_ratio`, `train_offset_8_ratio`, `train_offset_10_ratio`
- `train_missing_history_count`, `train_offset_sample_count`

Die Anteile beziehen sich auf alle verarbeiteten Trainingssamples einschließlich
Empty-History-Samples, deren Offset als Referenz 8 gezählt wird. DDP-Padding des
bestehenden DistributedSampler zählt als tatsächlich verarbeitetes Sample mit.
An Sequenzanfängen muss die empirische Gesamtverteilung deshalb nicht exakt der
konfigurierten Verteilung entsprechen. Bei anderen Configs wird auch ein separater
Referenzoffset außerhalb der Kandidatenliste gezählt.
Die Statistik wird auch nach stdout und über den bestehenden TensorBoard-Writer
geschrieben. Ohne aktives Trainingssampling werden diese Spalten nicht erzeugt.
Bestehende Loss-/IoU-Spalten bleiben unverändert. Das optionale bestehende
Sample-JSONL enthält zusätzlich `sampled_previous_offset`; die Basiseinstellung
`write_sample_logs:false` bleibt erhalten. Die erste Sampling-Trainingsbatch
pro Epoche loggt außerdem Shapes, Offsets, dt und Target-Shapes.

## HPC-Smoke-Test und vollständiger Start

Verwendet werden unverändert `submit_mos_p2_2gpu.sh` und `train_mos_p2_2gpu.slurm`:
Partition p2, ein Node, zwei GPUs (`gpu:2`), 16 CPUs, 64 GB RAM, zwei Tage,
vorhandenes Apptainer-Image und Torchrun mit zwei Prozessen. Der Wrapper erzeugt
einen eigenen Job-Logordner; es werden keine bestehenden Trainingslogs überschrieben.

Auf dem HPC-Login-Host:

```bash
cd /home_cu/s210891/LiDARSensorForecaster
bash submit_mos_p2_2gpu.sh \
  src/configs/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate_temporal_sampling.yaml \
  --max_batches 2 --epochs 1 \
  --log_root /hpc/logs/mos/temporal_sampling_smoke_final
```

Vollständiger Lauf, ohne Batch-/Epochenlimit und ohne Pretraining/Resume:

```bash
cd /home_cu/s210891/LiDARSensorForecaster
bash submit_mos_p2_2gpu.sh \
  src/configs/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate_temporal_sampling.yaml
```

Der Nutzer hat in seiner aktuellen Nachricht den anschließenden Start auf P2 mit
zwei GPUs angefordert; dies aktualisiert die ältere Vorgabe im Anhang, nur einen
Startbefehl vorzubereiten. Job-IDs und finale Prüfergebnisse stehen unten.

## Benchmark nach dem Training

Innerhalb des vorhandenen Apptainer-Containers auf einem zugewiesenen GPU-Knoten,
mit dem tatsächlichen neuen Run-Verzeichnis als `RUN_DIR`:

```bash
RUN_DIR=/hpc/logs/mos/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate_temporal_sampling/job-8307/MOS_temporal_e2e_lmnet_cb_gate_temporal_sampling_5_8_10_w1_20261109_135024
python3 tools/evaluate_mos_benchmark.py \
  --checkpoint "${RUN_DIR}/checkpoints/best_moving_iou.pt" \
  --dataset-root /hpc/SemanticKitti/dataset/sequences \
  --sequences 08 --mode both --device cuda \
  --postprocess knn --knn-k 5 --knn-search 5 \
  --knn-sigma 1.0 --knn-cutoff 1.0 \
  --moving-threshold 0.5 --history-init repeat_earliest \
  --output-dir "${RUN_DIR}/benchmark_seq08_off8_knn_threshold05"
```

Der Input bleibt 64×2048 XYZRI mit LMNet-Projektion und festem Offset 8.
`repeat_earliest` ist der vorhandene Benchmarkdefault. Für diesen Vergleich
werden keine zufälligen Offsets oder zusätzliche Robustheitsevaluationen 5/10
eingeführt. Das Benchmark-Ergebnis wird später mit ungefähr 61,14 % verglichen.

## Durchgeführte Prüfungen und Einschränkungen

Die neue Datei `tests/test_temporal_mos_offset_sampling.py` enthält zwölf Tests:
exakte Config-Vererbung, deaktivierter Pfad samt RNG/LMNet-Augmentation,
erzwungene Offsets mit echten Scan-Dateien und Pose-/Projektionsvergleich,
Sequenzanfänge/fehlende Scans/Posen/Sequenzgrenzen, empirische Verteilungen,
feste Validation/Test/Benchmarkeingaben, Multiworker 0/2 und Seedfolgen,
echtes cb_gate-Modell mit MOS-/Movable-Loss und Backward, Speichern/Resume/
alte Checkpoints, Inferenz auf dem Trainingssplit, ungültige Configs und
zwei CPU-Gloo-Ranks mit DistributedSampler, zwei Workern pro Rank,
echtem Modelltraining sowie unabhängig aus Sample-Logs geprüften DDP-Zählern.

Lokale CPU-Umgebung: PyTorch 2.14.1+cpu, Python 3.12. HPC-Container:
PyTorch 2.5.1+cu121, Python 3.10. Lokale Socket-Tests liefen außerhalb der
Sandbox, da deren Socketregeln sonst Multiprocessing/Gloo verhindern.
Syntaxprüfung, Imports, YAML-Validierung, `git diff --check` und Bash-Syntax
der vorhandenen P2-Skripte wurden geprüft.

Viele ältere Regressionstests verweisen auf inzwischen nach `src/configs/Archiv`
verschobene Originalconfigs. Die Regressionen wurden deshalb zusätzlich in
`/tmp/mos-variable-offset-regression` mit Kopien dieser vorhandenen Configs an
ihren alten Testpfaden ausgeführt. Die Configs im HPC-Projekt wurden dafür nicht
verschoben oder verändert. Zwei Residual-Tests scheitern auch mit dem unveränderten
HEAD-Code: `test_config_and_protocol` (archivierte Laufhyperparameter weichen von
der Testreferenz ab) und `test_distributed_accumulation_reference` (abweichendes
Referenz-Parameterupdate). Gegenprobe: `/tmp/variable-offset-prechange-residual.log`.
Die Residual-Pipeline ist nicht Teil des Samplingexperiments und wurde nicht repariert.

Der erste echte P2-Smoke-Test war Job 8305, abgeschlossen mit Exitcode 0.
Bei zwei A100-GPUs, 64×2048, Batch 8 pro Rank und zwei Trainings-/Validationsbatches
wurden 32 Trainingssamples gezählt: Offset 5=7, Offset 8=12, Offset 10=13,
fehlende Historie=1. MOS-/Movable-Losses waren endlich, Backward/Optimizerupdates
und Checkpoints erfolgreich. Die kurze Stichprobe erlaubt keine Aussage über
die spätere Modellqualität. Der Container meldet einen bestehenden DDP-
Gradientenstride-Hinweis ohne Laufabbruch.

Abschließender P2-Smoke-Test: Job **8306**, `COMPLETED`, Exitcode **0:0**,
Laufzeit 1:22 Minuten. Die erste Trainingsbatch bestätigte
`x=(8,2,5,64,2048)`, `valid=(8,2,64,2048)`, MOS-/Movable-Targets
`(8,64,2048)` und Offsets `[10,10,10,5,10,10,8,8]` mit passenden dt-Werten.
Die abschließend aus `epochs.csv` geprüften Zähler sind erneut 7/12/13 bei
32 Samples und einer fehlenden Historie. Aggregierte Trainingsloss 1,52178,
Movable-Loss 1,19295, Multitask-Loss 2,71473; Validation-Multitask-Loss 2,38441.
Die Validation verarbeitet ebenfalls 32 Samples, mit festem Offset 8.
Smoke-Logs und -Checkpoints:
`/hpc/logs/mos/temporal_sampling_smoke_final/MOS_temporal_e2e_lmnet_cb_gate_temporal_sampling_5_8_10_w1_20261109_134801`.

Finale lokale Regressionen mit dem finalen Code:

| Testgruppe | Ergebnis |
|---|---|
| Neue Offset-Sampling-Tests | 12/12 bestanden, einschließlich Gloo/DDP und Worker 0/2 |
| Temporal-/MOS-/MAE-Tests (`test_temporal*.py`) | 73 Tests: 69 bestanden, 2 übersprungen, 2 oben nachgewiesene bestehende Residual-Fehler |
| MOS-/Benchmark-/Loss-Regressionen (`test_mos_*.py`) | 50 Tests: 48 bestanden, 2 übersprungen |
| LMNet-Projektion und LMNet-MAE (`test_lmnet*.py`) | 20/20 bestanden |

Die neue Gruppe ist in den 73 Temporal-Tests bereits enthalten. Insgesamt sind
es somit 143 unterschiedliche Tests: 137 bestanden, 4 übersprungen und 2
nachgewiesene bestehende Residual-Fehler. Die gesamte Altsuite kann wegen dieser
bestehenden Fehler und der verschobenen Configpfade nicht als vollständig grün
bezeichnet werden. Änderungen an Modellen, Augmentation und Benchmark-Evaluator
wurden zusätzlich per Git-Diff ausgeschlossen.

Prüflogs, Smoke-CSV-Zusammenfassung und ein Source-Snapshot mit SHA-256-Manifest
liegen dauerhaft unter `/hpc/logs/mos/temporal_sampling_verification_20261109`.
Die bisherigen Referenzconfigs wurden weder überschrieben noch verschoben.

Der vollständige Lauf wurde nach erfolgreichen neuen Tests und GPU-Smoke-Test
mit dem oben dokumentierten unveränderten Startbefehl als Job **8307** auf
**p2 mit zwei GPUs** gestartet, Node `iwlhpc02.intern.th-ab.de`.
Keine CLI-Limits, kein Resume und kein Pretraining. Der vorhandene Wrapper
verwendet für dieses Training den neuen, separaten Container-Logroot
`/hpc/logs/mos/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate_temporal_sampling/job-8307`.
Slurm-stdout/stderr liegen auf dem Host unter
`/home_cu/s210891/logs/slurm/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate_temporal_sampling-8307.out`
bzw. `.err`.

Status auf dem Login-Host:

```bash
squeue -j 8307
tail -f /home_cu/s210891/logs/slurm/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate_temporal_sampling-8307.out
```

## Geänderte und neue Dateien

- Geändert: `src/temporal_mos_e2e_dataset.py`
- Geändert: `src/helper/temporal_e2e_input.py`
- Geändert: `tools/train_temporal_mos_e2e.py`
- Geändert: `tools/evaluate_temporal_mos_e2e.py`
- Neu: `src/helper/temporal_offset_sampling.py`
- Neu: `src/configs/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate_temporal_sampling.yaml`
- Neu: `tests/test_temporal_mos_offset_sampling.py`
- Neu: `TEMPORAL_MOS_VARIABLE_OFFSET_SAMPLING.md`
