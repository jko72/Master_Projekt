# Temporal E2E MOS: Variante A – Movable Auxiliary Supervision

## 1. Ausgangspunkt und wissenschaftliches Ziel

**Diese Variante übernimmt MF-MOS-inspirierte Movable-Supervision und Loss-Gewichtung, verwendet jedoch einen gemeinsamen Siamese-Backbone statt eines separaten Semantic Branch mit direkter Motion-Guided-Feature-Fusion.**

Die bestehende Pipeline verarbeitet t−8 und t als XYZRI, führt beide durch denselben
SalsaNext-Encoder, vergleicht die Bottleneck-Merkmale über Local Historical
Cross-Attention und dekodiert mit den aktuellen Skip-Features. `decoder.logits`
entscheidet weiterhin ausschließlich Static/Moving. Der zusätzliche 1×1-Conv-Head
klassifiziert dieselben 32 Decoderfeatures als Non-movable/Movable.

Geprüfte lokale Schnittstellen:

| Schnittstelle | Bisher | Erweiterung |
|---|---|---|
| `SalsaNextDecoder.forward` | MOS-Logits | optional `(logits, features)` |
| `SalsaNextTemporalE2E.forward` | MOS-Logits, optional Diagnostik | explizites `return_aux=True` |
| `TemporalMOSE2EDataset` | fünf Elemente | sechs Elemente bei Auxiliary |
| `TemporalE2EInputMixin` | Online-Projektion und eine Augmentationsziehung | gemeinsames `[N,2]`-Target |
| `MOSCrossEntropyLovasz` | CE/Lovasz separat | unverändert für beide Aufgaben wiederverwendet |
| `run_phase` | DDP-korrigierte MOS-CE, lokale Lovasz | separate Korrektur und Statistik für Movable |
| MAE-Transfer | strikte Übernahme außer MOS-Head | zusätzlich genau Movable-Head frisch |

Die Hypothese lautet, dass zusätzliche semantische Supervision die gemeinsamen
Merkmale verbessert. Eine tatsächliche Verbesserung der Point-MOS-Metriken ist
noch nicht nachgewiesen. Es wurden weder Residual-Auxiliary, TOP-Pretraining,
Offset-Sampling noch neue Feature-Fusion implementiert.

## 2. Geprüfte MF-MOS-Quellen und Abgrenzung

Autorenrepository: <https://github.com/SCNU-RISLAB/MF-MOS>, geprüfter Stand
`0c702445a39b978efc107cf7d0a2a33246f857ba`.
Die folgenden Dateien wurden tatsächlich abgerufen und gelesen:

- [modules/MFMOS.py](https://github.com/SCNU-RISLAB/MF-MOS/blob/0c702445a39b978efc107cf7d0a2a33246f857ba/modules/MFMOS.py): separate Featurepfade, MGA-Verbindungen, eigener Movable-Ausgang und Softmax.
- [modules/trainer.py](https://github.com/SCNU-RISLAB/MF-MOS/blob/0c702445a39b978efc107cf7d0a2a33246f857ba/modules/trainer.py): Aggregation von `content`, `1/(content+epsilon_w)`, ignorierte Klasse mit Gewicht null, NLL+Lovasz für beide Aufgaben, Summe ohne Aufgabenfaktor.
- [common/dataset/kitti/parser.py](https://github.com/SCNU-RISLAB/MF-MOS/blob/0c702445a39b978efc107cf7d0a2a33246f857ba/common/dataset/kitti/parser.py): gemeinsame Geometrie, beide projizierten Targets und Maskierung.
- [config/labels/semantic-kitti-mos.yaml](https://github.com/SCNU-RISLAB/MF-MOS/blob/0c702445a39b978efc107cf7d0a2a33246f857ba/config/labels/semantic-kitti-mos.yaml): `content` und `movable_learning_map`.
- [train_yaml/mos_coarse_stage.yml](https://github.com/SCNU-RISLAB/MF-MOS/blob/0c702445a39b978efc107cf7d0a2a33246f857ba/train_yaml/mos_coarse_stage.yml): `epsilon_w=0.001`, KNN 5/5/1/1.

Die eigene [Pipeline](https://github.com/jko72/Master_Projekt) wurde gegen die
tatsächlich lokal vorhandenen Dateien unter `LiDARSensorForecaster` geprüft.
Lokaler Code und die lokale HPC-Konfiguration waren die Implementierungsgrundlage.
Es wurde kein MF-MOS-Netzwerkcode kopiert; die Statistikfixture enthält die
faktischen Klassenhäufigkeiten und Zuordnungen für einen Offline-Quellentest.

| Komponente | Übernahme / bewusste Abweichung |
|---|---|
| Movable-Labelzuordnung | exakt MF-MOS, einschließlich 16/256 |
| Häufigkeitsaggregation und Gewichte | Originalstatistik, epsilon 0.001 |
| Aufgabenloss | gewichtete CE + Lovasz je Aufgabe, Aufgabenfaktor 1 |
| Raw-Logits | stabile CE statt NLL auf logarithmierten Softmax-Werten |
| Ignore | außerhalb der zwei gültigen Klassen als −1; MF-MOS hat zusätzlich einen Ignore-Ausgangskanal |
| Backbone | gemeinsamer Siamese-Encoder, Attention und Decoder |
| Semantic Branch / MGA-Fusion | keine separate Branch, keine direkte semantische Feature-Fusion |
| Eingang und Optimierung | bestehendes LMNet-nahes Temporal-Protokoll, nicht vollständige MF-MOS-Reproduktion |
| Inferenz | ausschließlich MOS-Head; keine Maskierung durch Movable-Probs |

CE entspricht NLL von log-softmax für dieselben gültigen Logits. Die
Zweiklassen-Ausgabe ohne trainierten Ignore-Ausgang ist dabei die vorgegebene
Pipeline-Konvention, keine vollständige numerische Reproduktion des
Dreikanal-Autorenmodells.

## 3. Exaktes Label-Mapping

`semantic_id = raw_uint32 & 0xFFFF`; Instanzbits werden entfernt.
`map_semantickitti_to_movable` ist unabhängig von `map_semantickitti_to_mos`.

| Target | Original-IDs |
|---|---|
| 1 Movable | 10, 11, 13, 15, 18, 20, 30, 31, 32, 251, 252, 253, 254, 255, 256, 257, 258, 259 |
| 0 Non-movable | 9, 16, 40, 44, 48, 49, 50, 51, 52, 60, 70, 71, 72, 80, 81, 99 |
| −1 Ignore | 0, 1, alle unbekannten IDs |

Ein geparktes Auto 10 ist MOS=0 / Movable=1. Ein fahrendes Auto 252 ist
MOS=1 / Movable=1. On-rails 16 bleibt Non-movable, moving-on-rails 256 ist
Movable. Die ursprüngliche MOS-Mapping-Funktion bleibt unverändert.

## 4. Originalstatistik und Klassengewichte

Für jede Originalklasse k wird deren `content[k]` anhand von
`movable_learning_map[k]` summiert. Fehlende Originalklassen wie 9 und 251
liefern keinen Statistikbeitrag, bleiben aber im Mapping enthalten.
Die Autorenklassen 1 und 2 werden hier 0 und 1.

| Klasse | Summierter content | `1/(content+0.001)` / YAML |
|---|---:|---:|
| Non-movable | 0.9331685037721359 | 1.07047068699 |
| Movable | 0.047646460740965804 | 20.55648005566 |

Diese festen Gewichte werden **nicht** aus Range-View-Pixeln neu geschätzt.
Ignore ist keine gültige Lossklasse. Zur Laufzeit liegen die CE-Gewichte wie im
bestehenden Training als FP32 vor (ca. 1.0704706907272339 und
20.556480407714844). Die CSV loggt diese tatsächlich verwendeten Werte.
`tests/reference/mfmos_movable_statistics.yaml` prüft die Aggregation offline.

## 5. Loss und Gradientfluss

Für Aufgabe q mit gültigen Pixeln V und Klassengewichten w:

\[
L_{CE,q}=\frac{\sum_{i\in V}w_{y_i}[-\log\mathrm{softmax}(z_i)_{y_i}]}{\sum_{i\in V}w_{y_i}},
\quad L_q=L_{CE,q}+L_{Lovasz,q}.
\]

\[
L_{Total}=L_{MOS}+1.0\,L_{Movable}.
\]

MOS-Gewichte bleiben `[1.0209899532210003, 296.43707590294014]`.
Beide Lovasz-Terme verwenden die bestehende `MOSLovaszSoftmax`-Implementierung:
`ignore_index=-1`, Gewicht 1, `classes=present`, `per_image=false`, explizite
FP32-Softmax/Sortierung/Dot-Operationen auch unter Autocast.
Leere Movable-Supervision erzeugt eine differenzierbare Null für CE und Lovasz.

Der Gradientenpfad ist:

```text
L_Movable → movable_head → 32 Decoderfeatures → Decoder-UpBlocks
                                            → aktuelle Encoder-Skips
                                            → Attention-Fusion
                                              → aktueller und historischer Shared Encoder
L_MOS     → decoder.logits → dieselben Decoderfeatures → dieselben gemeinsamen Komponenten
```

Es gibt kein `detach`. Ein isolierter Movable-Loss liefert keine Gradienten an
`decoder.logits`; ein gemeinsamer Loss trainiert beide Heads. Der Beitrag des
historischen Encoders hängt wie bisher von verfügbaren, gültigen Historyfeatures
und der Attention ab. Fehlende History bleibt null/invalid mit nominalem dt.

## 6. Projektion und synchrone Augmentation

Bei aktivierter Auxiliary wird die aktuelle `.label`-Datei einmal gelesen. Beide
Mappings werden als `[N,2]` gestapelt und gemeinsam an `raw_scans` übergeben.
Dessen einzige Geometrieziehung gilt für alle bereits ego-kompensierten Scans.
Punktlöschung erfolgt nur im aktuellen Scan und entlang `axis=0` für beide
Labelspalten. `[N]` funktioniert unverändert; falsche Dimensionen oder Längen
werden vor einer möglichen Zufallsziehung explizit verworfen.

`_project_lmnet_temporal` erzeugt genau ein aktuelles `Projection`-Objekt.
Beide Labelspalten werden mit dessen identischen Gewinnerindizes projiziert.
Bei Kollisionen kommen somit beide Labels vom selben Punkt. Range-View-Targets
werden als `[2,H,W]` gemeinsam gerollt, anschließend mit derselben aktuellen
Validitätsmaske auf −1 gesetzt. Unlabeled bleibt −1. Validation und Test haben
keine stochastische Augmentation. Weder die LMNet-Projektionsgleichungen noch
deren Kollisionsregeln wurden verändert.

Die Auxiliary-Variante projiziert auch Validierungstargets online; ein
MOS-Labelcache wird im LMNet-Pfad nicht gebraucht. Die deaktivierte Variante
behält ihren bisherigen Online-/Cache-Schalter. Der Legacy-Projektionspfad
unterstützt ebenfalls gemeinsame Targetindizes und synchrone Range-View-Rotation;
die neue Versuchskonfiguration benutzt ausschließlich LMNet.

## 7. Modell- und Dataset-Verträge

Für die neue Konfiguration pro Sample:

```text
x         [T=2,5,64,2048]  Kanäle x,y,z,range,intensity
valid     [T=2,64,2048]    bool
dt        [T=2]           Sekunden [-0.8, 0]
y_mos     [64,2048]        int64, {-1,0,1}
y_movable [64,2048]        int64, {-1,0,1}
```

Ohne Auxiliary: `(x, valid, dt, y_mos, meta)`.
Mit Auxiliary: `(x, valid, dt, y_mos, y_movable, meta)`.
Test 11–21 liefert `None` für beide Targets und benötigt keine `.label`-Dateien.
`collate_mos` unterstützt beide Tupel auch ohne Ground Truth.

```python
mos_logits = model(x, valid, dt)  # [B,2,H,W], Movable-Head wird nicht berechnet
mos_logits, movable_logits = model(x, valid, dt, return_aux=True)
# Mit bisheriger Attention-Diagnostik:
(mos_logits, movable_logits), diagnostics = model(
    x, valid, dt, return_aux=True, return_diagnostics=True)
```

`decoder(..., return_features=True)` liefert `(mos_logits, features[B,32,H,W])`.
Der Decoder wird pro Modell-Forward einmal ausgeführt. `return_aux=True` bei
deaktivierter Auxiliary wird explizit abgelehnt. Der normale Forward und das
bisherige `return_diagnostics=True` behalten ihren Vertrag.

Der neue Head wird auf CPU in einem isolierten RNG-Kontext mit Seed 1729 erzeugt.
Die bestehende Scratch-Initialisierung sowie die anschließende CPU-RNG-Folge
bleiben bitgleich. Es werden 66 Parameter ergänzt: insgesamt 7.075.972 statt
7.075.906. Weitere Aktivierungen und Loss-Sortierung erhöhen den Trainingsspeicher;
der tatsächliche GPU-Bedarf muss auf dem Zielsystem gemessen werden.

## 8. DDP, Akkumulation und Logging

Für jede Aufgabe wird separat die lokale Gewichtssumme D_r und deren globale
Summe D ermittelt. Vor dem DDP-Gradientenmittel wird CE mit
`world_size * D_r / D` korrigiert. Bei D=0 wird der Nenner sicher begrenzt;
der differenzierbare Null-Loss hält alle Headparameter im Graphen.
Lovasz bleibt wie in der Baseline lokal pro Rank und wird durch DDP gemittelt.
Die globale Sortierung über alle Ranks wird nicht approximiert oder behauptet.

Gradient Accumulation mittelt wie bisher die global normalisierten
**Microbatch-Losses**, einschließlich einer kürzeren letzten Gruppe. Dies ist
bei unterschiedlichen Gewichtssummen nicht identisch mit einer einzigen CE über
alle Pixel des gesamten Akkumulationsfensters. Diese Baseline-Semantik bleibt
erhalten. SGD, Warmup, Decay, Clipping und AMP/Scaler werden weiterverwendet.

Es gibt keine Kommunikation im Modell-Forward. Zusätzliche Nenner-Allreduces
finden nur beim Training statt. Validation benutzt das entpackte Modell mit zuvor
synchronisierten Buffern und reduziert Statistiken erst am Ende. Ungleich lange
oder leere Validierungsshards benötigen keine schrittweisen Collectives.

Bestehende CSV-Felder bleiben erhalten: `loss`, `loss_ce`, `loss_lovasz`,
`loss_total` bezeichnen weiterhin **MOS**, ebenso Moving-IoU/F1 usw.
Zusätzlich je `train_`/`val_`:

- `movable_loss_ce`, `movable_loss_lovasz`, `movable_loss_total`
- `multitask_loss_total` = MOS-Gesamtloss + Aufgabenfaktor × Movable-Gesamtloss
- `movable_iou`, `movable_f1`, `movable_precision`, `movable_recall`
- `movable_valid_pixels`
- `movable_non_movable_class_weight`, `movable_class_weight`
- `mos_static_class_weight`, `mos_moving_class_weight`, `parameter_count`
- bestehendes `peak_vram_bytes` für jede Phase

CE-Statistiken werden mit ihren jeweiligen Gewichtssummen über die Epoche
aggregiert; Lovasz wird über lokale Batches gemittelt. TensorBoard behält die
bisherigen Tags und ergänzt `train/...` sowie `val/...`. Parameterzahl wird beim
Start ausgegeben; der Optimizer enthält überprüft alle trainierbaren Parameter.

## 9. Checkpoints, Resume und MAE-Transfer

MOS-only-Modelle haben exakt ihre bisherigen State-Keys. Alte MOS-Checkpoints
werden mit deaktivierter Auxiliary weiterhin `strict=True` geladen.
Auxiliary-Modelle ergänzen ausschließlich `movable_head.weight` und
`movable_head.bias`; alle bisherigen Keys einschließlich `decoder.logits.*`
bleiben bestehen. Neue Checkpoints enthalten vollständiges Modell, Optimizer,
Scheduler, Scaler, RNG-Zustände, Config und `auxiliary_metadata`.

`--resume` vergleicht den vollständigen aufgelösten Auxiliary-Block zusätzlich
zum bestehenden Daten-/Modell-/Optimierungsprotokoll. Deaktivieren, andere
Gewichte, Lossoptionen oder Aufgabenfaktor werden zurückgewiesen. Es gibt keine
stille Konvertierung eines Auxiliary-Trainings in MOS-only-Training.

MAE-Transfer prüft Quellformat, Metadaten, exakte erlaubte Keymenge, Shapes,
Dtypes und endliche Werte vor der Mutation. Encoder, Attention und Decoder
werden übernommen; genau MOS- und Movable-Head bleiben frisch initialisiert.
Das zusammengeführte State-Dict wird weiterhin `strict=True` geladen.

## 10. Neue Konfiguration und unveränderte Baseline

Neue Datei: `src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_movable.yaml`.
Quelle ist die tatsächlich lokal vorhandene
`src/configs/mos_semanticKitti_temporal_e2e_lmnet_v1_p2_2gpu_online.yaml`,
Run `MOS_temporal_e2e_off8_fullres_lmnet_v1_p2_2gpu_online_100ep`.
Diese Baseline-Datei ist lokal über `.git/info/exclude` ausgeschlossen. Sie wurde
nicht verändert. Eine Testfixture unter
`tests/reference/temporal_mos_lmnet_online_baseline.yaml` bewahrt ihr vollständiges
Protokoll für den Vergleich auch in frischen Checkouts. Die neue Konfiguration
wurde explizit für die Git-Diff-Ansicht aufgenommen.

Ein automatisierter Vergleich erlaubt ausschließlich Auxiliary-Block und neuen
Run-Namen als Unterschiede. Insbesondere bleiben erhalten:
64×2048, FOV +3/−25°, Range 0/null, `lmnet`/`lmnet_v1`, XYZRI,
`paper_normalization`, `[8]`, 10 Hz, Split 00–07/09/10, 08, 11–21;
Dropout .1, 8 Heads, Radius 1, Compare 64, `bottleneck_only`;
MOS CE+Lovasz 1/present/batch; SGD .01/.9/.0001;
LMNet-Warmup 1, Decay .99, 100 Epochen, Seed 42, deterministic,
AMP false, Batch 8 pro GPU, Accumulation 1, zwei GPUs/global 16;
`checkpoint_metric=mean_iou`, `pretrained_path=null`, LMNet-Augmentation.
Auch `num_workers=0`, `clip_grad_norm=0`, `write_sample_logs=false` und die
vorhandenen Label-/Logging-Einstellungen bleiben übernommen.

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

Alle fünf Unterbereiche werden strikt validiert; unbekannte Keys auch in
Unterblöcken sind Fehler. `enabled` und `lovasz_per_image` sind Booleans,
`task=movable` für diese Variante (die separate Variante B unterstützt zusätzlich `residual`), `loss_weight` endlich und ≥0,
Klassen-Gewichte genau zwei positive endliche Zahlen, Ignore ausschließlich −1.
Auxiliary-Loss ist ausschließlich `ce_lovasz` mit Lovasz-Faktor 1;
`lovasz_classes` erlaubt wie bisher `present`/`all`.
`use_auxiliary_output` muss `false` sein. Es existiert kein Schalter zum
Umlenken der offiziellen MOS-Inferenz. Ohne Block oder mit `enabled:false`
werden keine zusätzlichen Modellparameter und Targets angelegt.

## 11. Reproduzierbare Training- und Evaluationsbefehle

Im Repository, mit dem üblichen PyTorch-Environment:

```bash
# Volle synthetische Auflösung, beide Heads, ein Backward; keine Logdateien im Repo.
python tools/train_temporal_mos_e2e.py \
  --cfg_path src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_movable.yaml \
  --device cpu --dry-run

# Auf dem HPC im bestehenden Container: realer Smoke vor dem vollständigen Run.
# Logroot separat halten; --max_batches begrenzt sowohl Training als auch Validation.
python tools/train_temporal_mos_e2e.py \
  --cfg_path src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_movable.yaml \
  --device cuda --epochs 1 --batch_size 1 --num_workers 0 --max_batches 1 \
  --log_root /hpc/logs/mos/aux_movable_smoke

# 100 Epochen, bestehender P2-Launcher, 2 GPUs, Batch8/GPU, global16.
bash submit_mos_p2_2gpu.sh \
  src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_movable.yaml
```

Der vorhandene Launcher bindet `${HPC_ROOT}` nach `/hpc`, startet zwei
`torch.distributed.run`-Prozesse und legt einen neuen Job-Logordner an.
Für Konfigurationsname und Job-ID in den Slurm-Logs den Submit-Wrapper verwenden;
er erzeugt `hpc/logs/slurm/<Konfigurationsname>-<Job-ID>.out/.err`.
Es wurde kein HPC-Job abgeschickt.

```bash
# CKPT durch den neuen best_mean_iou.pt ersetzen; keine Baseline-Datei überschreiben.
CKPT=/hpc/logs/mos/NEW_RUN/checkpoints/best_mean_iou.pt
python tools/evaluate_temporal_mos_e2e.py \
  --checkpoint "$CKPT" --split val --device cuda --batch_size 1

# Offizielle Point-MOS-Auswertung: Originalpunkte, Validation08, identisches KNN.
python tools/evaluate_mos_benchmark.py \
  --checkpoint "$CKPT" \
  --dataset-root /hpc/SemanticKitti/dataset/sequences \
  --sequences 08 --mode point --postprocess knn \
  --knn-k 5 --knn-search 5 --knn-sigma 1.0 --knn-cutoff 1.0 \
  --moving-threshold 0.5 --device cuda \
  --output-dir /hpc/logs/mos/NEW_RUN/point08_knn5_aux_movable
```

Der Point-Evaluator baut das Modell anhand der Checkpoint-Config und lädt es
strikt. Der Default-Forward liefert weiterhin nur MOS. Optional aktiviert
`--movable-diagnostics` genau einen Forward mit `return_aux=True`, führt die
MOS-Baseline unverändert (inklusive KNN) aus und wendet erst auf deren
Punktvorhersage den Movable-Filter an. Die Baseline bleibt in
`point_metrics.csv`; die zusätzlichen Ergebniszeilen stehen in
`movable_diagnostics.csv`.
Pose-/Mapping-/Threshold-/KNN- und `.label`-Exportlogik bleiben bestehen.
Der Range-View-Evaluator deaktiviert ausschließlich für seinen Dataset-Aufbau
den Auxiliary-Block; das Modell wird davor mit der vollständigen Originalconfig
gebaut und strikt geladen. Auch `--split=train` bleibt augmentationsfrei.
Separate Movable-Analysen verwenden ausdrücklich `return_aux=True` und sind
kein Bestandteil der offiziellen MOS-Auswertung.

### Nachgelagerte Movable-Diagnose (P1, eine GPU)

Der Checkpoint wird aus seiner gespeicherten Config rekonstruiert. Ein
Residual-Checkpoint, ein Modell ohne aktivierten Movable-Head oder fehlende
originale `.label`-Ground-Truth wird deshalb vor der Auswertung abgewiesen.
Die Diagnose nutzt die direkte Originalpunkt-zu-Pixel-Zuordnung für den
kontinuierlichen Movable-Score, während MOS weiterhin mit KNN rückprojiziert
wird. Punkte ohne gültigen Movable-Score bleiben unverändert.

```bash
CKPT=/hpc/logs/mos/mos_semanticKitti_temporal_e2e_lmnet_aux_movable/job-6967/MOS_temporal_e2e_lmnet_off8_aux_movable_mfmos_w1_20261031_190312/checkpoints/best_moving_iou.pt

# Smoke-Test, eigener Output-Ordner
python3 tools/evaluate_mos_benchmark.py --checkpoint "$CKPT" \
  --dataset-root /hpc/SemanticKitti/dataset/sequences --sequences 08 --mode point \
  --postprocess knn --knn-k 5 --knn-search 5 --knn-sigma 1.0 --knn-cutoff 1.0 \
  --knn-chunk-size 32768 --movable-diagnostics --max-frames 10 \
  --output-dir /hpc/logs/mos/movable_diag_smoke_$(date +%Y%m%d_%H%M%S)

# Vollständige Val08-Auswertung (4.071 Scans), eigener Output-Ordner
python3 tools/evaluate_mos_benchmark.py --checkpoint "$CKPT" \
  --dataset-root /hpc/SemanticKitti/dataset/sequences --sequences 08 --mode point \
  --postprocess knn --knn-k 5 --knn-search 5 --knn-sigma 1.0 --knn-cutoff 1.0 \
  --knn-chunk-size 32768 --movable-diagnostics \
  --output-dir /hpc/logs/mos/movable_diag_val08_$(date +%Y%m%d_%H%M%S)

# Als P1-Job: immer den Submit-Wrapper verwenden, damit HPC_ROOT, Container-
# Mounts und die Logpfade wie beim Training gesetzt werden.
bash submit_evaluate_mos_movable_diagnostics_p1_1gpu.sh \
  /home/koehlbrandt/hpc/logs/mos/mos_semanticKitti_temporal_e2e_lmnet_aux_movable/job-6967/MOS_temporal_e2e_lmnet_off8_aux_movable_mfmos_w1_20261031_190312/checkpoints/best_moving_iou.pt \
  /hpc/logs/mos/movable_diag_val08_$USER
```

`movable_diagnostics.csv` enthält pro Schwelle `removed_mos_false_positives`
und `lost_mos_true_positives` sowie `removed_aux_nonmovable_points` und
`removed_aux_movable_points`. Ein hoher erster bei niedrigem zweiten Wert zeigt
gezielt entfernte MOS-Falschpositive; ein hoher `lost_mos_true_positives` oder
`removed_aux_movable_points` zeigt den Preis in echten Treffern. Die Konsole
meldet zusätzlich die unveränderte Baseline, Schwelle 0,5 und den besten
explorativen Val08-Sweep-Punkt. Diese Val08-Wahl vor einem unabhängigen Test
festschreiben.

## 12. Automatisierte Tests und tatsächlich ausgeführte Checks

Neue Suite: `tests/test_temporal_mos_aux_movable.py`, neun Tests mit zahlreichen
Unterprüfungen: exaktes Mapping inklusive Instanzbits und unbekannter Klassen,
MF-MOS-Statistikaggregation, Baseline-Konfigurationsvergleich, strikte Validierung,
Projektionskollisionen, Invalid-Pixel, `[N,2]`-Punktlöschung, 1D-Kompatibilität,
gemeinsame RNG-Ziehung, Train/Val/Test-Verträge, fehlende History,
Range-View-Rotation, exakt gemeinsame Feature-Tensoridentität, MOS-only-Forward,
Diagnostik, Parameterkeys, initiale RNG-Parität, Gradienten, CE-/Lovasz-Komponenten,
leere Supervision, strikter MAE-Transfer, Training/Validation, alle drei
Checkpoints, Neuladen, MOS-Evaluation und erfolgreicher/inkompatibler Resume.

Der Zwei-Prozess-Gloo-Test vergleicht den tatsächlichen SGD-Update mit einer
Referenz aus global gewichteter CE pro Aufgabe und dem Mittel lokaler
Lovasz-Terme. Er testet unterschiedliche Targetmasken, einen lokal und einen
global leeren Movable-Microbatch, Accumulation 2, eine verkürzte letzte Gruppe,
ungleich lange Validation inklusive leerem Rank sowie zwei Backwards des echten
Temporal-Modells ohne fehlende Gradienten/DDP-unused-Parameter.

Exakte lokale Testbefehle (Projektwurzel):

```bash
MPLCONFIGDIR=/tmp/mos-mpl OMP_NUM_THREADS=2 /tmp/mos-lovasz-venv/bin/python \
  -m unittest discover -s tests -p 'test_temporal_mos_aux_movable.py' -v
MPLCONFIGDIR=/tmp/mos-mpl OMP_NUM_THREADS=2 /tmp/mos-lovasz-venv/bin/python \
  -m unittest discover -s tests -p 'test_mos_*.py' -v
MPLCONFIGDIR=/tmp/mos-mpl OMP_NUM_THREADS=2 /tmp/mos-lovasz-venv/bin/python \
  -m unittest discover -s tests -p 'test_lmnet*.py' -v
MPLCONFIGDIR=/tmp/mos-mpl OMP_NUM_THREADS=2 /tmp/mos-lovasz-venv/bin/python \
  -m unittest discover -s tests -p 'test_temporal*.py' -v
```

DDP und multiprocessing benötigen lokale Socketzugriffe. Die Sandbox blockierte
zunächst Gloo; die betreffenden Tests wurden mit Freigabe außerhalb der Sandbox
ausgeführt. Während der Implementierung erkannte Syntax- und Testfixturefehler
wurden korrigiert; maßgeblich sind die abschließenden Läufe.

Ergebnisse am 31.10.2026, PyTorch `2.14.1+cpu`:

- Neue Auxiliary-Suite: **9 Tests bestanden**.
- Vorhandene MOS-Suiten: **47 Tests, OK, 2 übersprungen** (GPU/AMP ohne CUDA).
- LMNet-/MAE-Suiten: **20 Tests bestanden**.
- Temporal-Suiten einschließlich Auxiliary: **35 Tests bestanden**.
  Zusammen mit MOS und LMNet: **102 Tests, 100 bestanden, 2 CUDA-Skips**.
- CPU-Synthetic mit unveränderter neuer YAML: `[1,2,5,64,2048]`,
  beide Heads, vollständiger Backward/Optimizer-Schritt, bestanden;
  Gesamtloss 2.68656, ein Versuch, Gradienten aller Parameter endlich/vorhanden.
- Realer SemanticKITTI-Trainings- und Validierungsbatch: je ein Batch, Batchgröße1,
  volle 64×2048, Offset8, CPU, keine Protokolländerungen außer lokalem Daten-/Logpfad
  und Smoke-Limits. 19.130 Training- und 4.071 Validation-Frames indexiert.
  Training: MOS CE .760796, Lovasz .499392, Movable CE .741640,
  Lovasz .497414, Multi-Task 2.499242, 81.551 gültige Movable-Pixel.
  Validation: MOS 1.202798, Movable 1.190039, Multi-Task 2.392838,
  94.194 gültige Movable-Pixel. Alle Parameter im Optimizer, 7.075.972 Parameter.
- Realen Checkpoint im MOS-Range-View-Evaluator strikt neu geladen und einen
  Validation-Batch erfolgreich ausgewertet: MOS-Loss 1.202798.
- Realen Checkpoint im unveränderten Point-KNN-Evaluator für einen Originalscan
  aus Sequenz08 erfolgreich ausgewertet, KNN 5/5/1/1, Threshold .5.
  Moving-IoU 0 nach nur einem Trainingsschritt ist kein wissenschaftliches Ergebnis.
- Keine nutzbare CUDA-GPU (`nvidia-smi` ohne Treiberverbindung), kein GPU-Smoke,
  keine CUDA-AMP-/NCCL-/GPU-VRAM-Aussage. CPU-VRAM-Log ist erwartungsgemäß 0.

Lokale Smoke-Konfiguration wurde ausschließlich unter `/tmp` erstellt:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path('src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_movable.yaml').read_text())
cfg['dataset_path'] = str(Path('../SemanticKitti/dataset/sequences').resolve())
cfg['mos_log_params']['log_root'] = '/tmp/aux-movable-real-runs'
cfg['mos_log_params']['use_tensorboard'] = False
Path('/tmp/aux-movable-real.yaml').write_text(yaml.safe_dump(cfg))
PY
MPLCONFIGDIR=/tmp/mos-mpl OMP_NUM_THREADS=2 /tmp/mos-lovasz-venv/bin/python \
  tools/train_temporal_mos_e2e.py --cfg_path /tmp/aux-movable-real.yaml \
  --device cpu --epochs 1 --batch_size 1 --num_workers 0 --max_batches 1
```

Tatsächlich ausgeführte Reload-/Point-Smokes auf dem gespeicherten realen Run:

```bash
CKPT=/tmp/aux-movable-real-runs/MOS_temporal_e2e_lmnet_off8_aux_movable_mfmos_w1_20261031_185048/checkpoints/last.pt
MPLCONFIGDIR=/tmp/mos-mpl OMP_NUM_THREADS=2 /tmp/mos-lovasz-venv/bin/python \
  tools/evaluate_temporal_mos_e2e.py --checkpoint "$CKPT" \
  --device cpu --split val --max_batches 1
MPLCONFIGDIR=/tmp/mos-mpl OMP_NUM_THREADS=2 /tmp/mos-lovasz-venv/bin/python \
  tools/evaluate_mos_benchmark.py --checkpoint "$CKPT" \
  --dataset-root /home/koehlbrandt/hpc/SemanticKitti/dataset/sequences \
  --sequences 08 --mode point --postprocess knn \
  --knn-k 5 --knn-search 5 --knn-sigma 1.0 --knn-cutoff 1.0 \
  --moving-threshold 0.5 --device cpu --max-frames 1 \
  --output-dir /tmp/aux-movable-point-smoke
```

Tatsächliche Artefakte: `/tmp/aux-movable-real-runs/` und
`/tmp/aux-movable-point-smoke/`. Testprotokolle:
`/tmp/aux-movable-tests.log`, `/tmp/aux-mos-regression.log`,
`/tmp/aux-lmnet-regression.log`, `/tmp/aux-temporal-regression.log`,
`/tmp/aux-movable-dry-run.log`, `/tmp/aux-movable-real.log`,
`/tmp/aux-movable-real-eval.log`, `/tmp/aux-movable-point.log`.
Bestehende Experimente und Ergebnisdateien wurden nicht verändert.

## 13. Experimentmatrix und verbleibende Verifikation

| Experiment | Initialisierung | Movable | Aufgabenfaktor | Zweck |
|---|---|---|---:|---|
| dokumentierte LMNet-nahe Baseline | Scratch Seed42 | aus | – | bestehende Referenz |
| Variante A (diese YAML) | identischer Scratch Seed42 | MF-MOS-Mapping/Gewichte | 1.0 | Hauptvergleich |
| spätere Replikation | weitere identische Seeds je Paar | aus / an | 1.0 | Streuung und Robustheit |

Dokumentierte Baseline: Point-Moving-IoU **59,2235 %**, F1 **74,3904 %**,
Precision **77,8155 %**, Recall **71,2542 %**, 4.071 Originalscans von08,
KNN 5/5/1/1, Threshold .5. Diese Werte sind die bereitgestellte Referenz,
keine in dieser Änderung erneut ausgeführte Baseline-Evaluation.

Vor einer Ergebnisbehauptung erforderlich: 100 Epochen auf zwei Ziel-GPUs,
Batch8/GPU/global16, endliche Losses und VRAM prüfen; danach
`best_mean_iou.pt` über alle 4.071 Scans mit identischer Point-KNN-Auswertung
vergleichen. `best_moving_iou.pt` und `last.pt` bleiben ebenfalls erhalten.
Zusätzliche Seeds und die getrennte Auswertung von Movable-Verwechslungen sind
sinnvolle Folgeanalysen. Ein einzelner Seed isoliert den Effekt nicht statistisch.

Methodische Grenzen: grobe semantische Beweglichkeit ist keine aktuelle Bewegung;
Klasse16/256 ist bewusst asymmetrisch; Autoren-Punktstatistik unterscheidet sich
von der projizierten Pixelverteilung; gemeinsame Aufgaben können konkurrierende
Gradienten erzeugen; Lovasz bleibt ranklokal; Validation während des Trainings
misst Range-View-MOS, während der wissenschaftliche Endpunkt Point-MOS ist.
Die zusätzliche Supervision enthält keine neue Inferenzinformation.

## 14. Dateiübersicht

Neu:
`TEMPORAL_MOS_AUX_MOVABLE_MFMOS.md`,
`src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_movable.yaml`,
`src/helper/mos_auxiliary.py`,
`tests/test_temporal_mos_aux_movable.py`,
`tests/reference/mfmos_movable_statistics.yaml`,
`tests/reference/temporal_mos_lmnet_online_baseline.yaml`.

Geändert:
`src/helper/mos_label_helper.py`, `src/helper/mos_augmentation.py`,
`src/helper/mos_online.py`, `src/helper/temporal_e2e_input.py`,
`src/helper/temporal_e2e_transfer.py`, `src/temporal_mos_e2e_dataset.py`,
`src/mos_models/salsanext_parts.py`, `src/mos_models/salsanext_temporal_e2e.py`,
`src/mos_models/builder.py`, `tools/train_temporal_mos_e2e.py`,
`tools/evaluate_temporal_mos_e2e.py`.

## 15. Kompakter Current-Bottleneck-Zweig mit optionaler semantischer Rückkopplung

Stand 04.11.2026. Diese Erweiterung wurde ausschließlich im bereits ausgecheckten
HPC-Branch `hpc/p1-ddp-mos` auf Basis von `db4a8cb` implementiert. Es war kein
Branchwechsel nötig. Im Repository und den geprüften übergeordneten Verzeichnissen
wurde keine `AGENTS.md` gefunden. Vorhandene lokale Änderungen an der
Movable-Diagnose, deren Tests und P1-Launchern wurden bewahrt.

Verbindliche Vorlage nach ausdrücklicher Bestätigung des Nutzers ist die lokale
`src/configs/mos_semanticKitti_temporal_e2e_lmnet_aux_movable.yaml`.
Sie liegt nicht im Git-Tree des HPC-Branches, sondern ist lokal über
`.git/info/exclude` ausgeschlossen. Ihr SHA256 vor und nach der Bearbeitung:

```text
af77d1f0e7c4ceb2cd7fd395da1e354ca215e43d066b71b3b9cb24662d6d9bca
```

Die Vorlage wurde nicht geändert. Die beiden neuen YAMLs wurden direkt aus ihr
abgeleitet und bleiben wie die übrigen HPC-Konfigurationen lokal ausgeschlossen;
sie wurden nicht gestaged oder gepusht. Es wurde keine andere Basiskonfiguration
als Ersatz herangezogen.

**Die früheren Abschnitte und Testergebnisse beschreiben die Legacy-Variante.**
Für die neue kompakte Architektur wurden in dieser Implementierungsumgebung
keine Tests, Dry-Runs, Forward-/Backward-Läufe oder Trainings ausgeführt und
keine Jobs eingereicht. Die unten beschriebenen Ausführungsprüfungen sind nur
vorbereitet und müssen auf dem HPC stattfinden.

### Drei klar getrennte Modellkonfigurationen

| Variante | `movable_branch` | `semantic_fusion` | Movable-Features | Direkte semantische MOS-Rückkopplung |
|---|---|---|---|---|
| Unveränderte Referenz / Legacy | `shared_decoder` oder fehlend | `none` oder fehlend | letzte 32 MOS-Decoderfeatures | keine |
| Kompakte Ablation ohne Gate | `current_bottleneck` | `none` | aktueller Bottleneck und aktueller Viertelauflösungs-Skip | keine |
| Kompakte Ablation mit Gate | `current_bottleneck` | `bottleneck_gate` | exakt derselbe kompakte Zweig | vor dem MOS-Decoder |

`semantic_fusion: none` wählt nicht den Legacy-Head zurück, sondern schaltet
bei `current_bottleneck` ausschließlich das Gate ab. Das bestehende
`fusion: bottleneck_only` bezeichnet weiterhin die **zeitliche** Attention-Fusion.

Neue vollständige Konfigurationen:

```text
src/configs/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_none.yaml
src/configs/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate.yaml
```

Gemeinsamer Modellblock, zusätzlich zu allen bisherigen Referenzfeldern:

```yaml
mos_model_params:
  # name, in_channels, num_classes, dropout, attention_heads, radius,
  # compare_channels und fusion sind aus der vollständigen Vorlage übernommen.
  movable_branch: current_bottleneck
  semantic_fusion: none  # zweite YAML: bottleneck_gate
```

Die beiden Dateien unterscheiden sich ausschließlich in `semantic_fusion` und
`mos_log_params.run_name`. Die Run-Namen lauten:

```text
MOS_temporal_e2e_lmnet_off8_movable_cb_none_mfmos_w1
MOS_temporal_e2e_lmnet_off8_movable_cb_gate_mfmos_w1
```

Alle anderen Einstellungen bleiben identisch zur lokalen Vorlage, insbesondere
Scratch (`pretrained_path: null`), Seed 42, Worker 0, die Splits, XYZRI,
`paper_normalization`, Historie `[8]`, LMNet-Augmentation, 64×2048, Batch 8 pro
GPU, Accumulation 1, zwei DDP-Ranks/global 16, SGD, Scheduler, 100 Epochen,
MOS- und Movable-Gewichte sowie `checkpoint_metric: mean_iou`.

### Datenfluss und tatsächliche Kanal-/Rastergrößen

Die tatsächliche Encoderdefinition wurde geprüft: `res_block5` liefert 256
Bottleneck-Kanäle bei H/16×W/16. `current_skips[2]` ist `down2b`, also das
256-Kanal-Ergebnis von `res_block3` **vor** dessen Pooling bei H/4×W/4.
Die Kanalzahlen für die neuen Projektionen werden aus den vorhandenen
`encoder.res_block5.conv1.out_channels` und
`encoder.res_block3.conv1.out_channels` gelesen, nicht aus einem angenommenen
externen SalsaNext-Modell. Rastergrößen stammen aus den Tensor-Shapes.

```text
x[t-8], x[t] → ein gemeinsamer Encoder-Forward über B*T
                      │
                      ├─ encoded[:, -1] vor temporaler Attention
                      │      → Conv1×1(256→64) + GELU = S_t
                      │           ├─ bilinear auf current_skips[2].shape[-2:]
                      │           │   + concat mit Conv1×1(256→32)(current_skips[2])
                      │           │   → Conv3×3(96→32) + GELU
                      │           │   → Conv1×1(32→2)
                      │           │   → bilinear auf H×W = Movable-Logits
                      │           └─ optional P = Conv1×1(64→256) → tanh-Gate
                      │
                      └─ bestehende Local Historical Attention = F_temporal
                                 → optionales Gate mit S_t
                                 → unveränderter MOS-Decoder und decoder.logits
```

Bei 64×2048 hat S_t die Form `[B,64,4,128]`, der ausgewählte Skip
`[B,256,16,512]`, die semantische Logit-Zwischenstufe `[B,2,16,512]` und
Movable nach Interpolation `[B,2,64,2048]`. Nur die zwei **rohen Logits** werden
auf volle Auflösung interpoliert. Alle bilinearen Interpolationen verwenden
`align_corners=False`. Es gibt keinen zweiten Encoder und keinen vollständigen
zweiten SalsaNext-Decoder.

Die zusätzlichen semantischen Module besitzen rechnerisch 52.418 Parameter.
Das Gate ergänzt weitere 16.640 Parameter. Dies sind aus den Conv-Definitionen
abgeleitete Größen, keine in dieser Umgebung ausgeführten Modellmessungen.
Die tatsächliche Gesamtparameterzahl wird vom bestehenden Trainer ausgegeben
und in `train_parameter_count`/`val_parameter_count` protokolliert.

### Exakte Gate-Definition, Gradienten und Inferenz

Mit der vorhandenen aktuellen Bottleneck-Maske M gilt:

```text
F_out = F_temporal + 0.1 * F_temporal * tanh(P(S_t)) * M
```

P wird einschließlich Bias mit null initialisiert. Der zusätzliche Term ist
zunächst null, das Gate startet als Identität. Der erste MOS-Backward kann P
trainieren. Der zusätzliche MOS-Gradient nach S_t ist bei exakt null gesetztem
P zunächst null; bei nichttrivialen P-Gewichten kann er bis zum Semantikadapter
fließen. Es gibt kein `detach()` an S_t.

S_t ist dasselbe Tensorobjekt für Movable-Head und Gate. Der kompakte
Movable-Loss trainiert den Semantikadapter, den aktuellen Skip-Pfad und den
geteilten Encoder. Anders als beim Legacy-Head läuft sein direkter Gradient
nicht durch die zeitliche Attention oder den MOS-Decoder. Diese werden durch
MOS trainiert. Mit aktivem Gate erhält der Semantikadapter nach Entwicklung
nichttrivialer Gate-Gewichte zusätzlich MOS-Gradienten.

Adapter und kompakter Movable-Head werden in einem lokalen CPU-RNG-Kontext mit
Seed 1729 erzeugt. Das optionale Gate verwendet einen getrennten Kontext mit
Seed 1730 und wird danach auf null gesetzt. So sollen die gemeinsamen Gewichte,
der vollständige neue Movable-Zweig und der nachfolgende RNG-Zustand zwischen
den beiden Ablationen identisch sein. Der alte Legacy-Head behält seine bisherige
Initialisierung und Parameterreihenfolge. Diese Eigenschaften werden von den
vorbereiteten Prüfungen überprüft; sie wurden hier nicht ausgeführt.

Die bisherigen APIs bleiben erhalten:

```python
mos = model(x, valid, dt)
mos, movable = model(x, valid, dt, return_aux=True)
(mos, movable), diagnostics = model(
    x, valid, dt, return_aux=True, return_diagnostics=True)
```

Bei `none` gelangt F_temporal unmittelbar in den MOS-Decoder. Der gesamte
Semantikpfad kann bei einem reinen MOS-Forward entfallen. Bei `bottleneck_gate`
werden S_t und das Gate **auch in der regulären Inferenz** berechnet; nur der
kompakte Movable-Decoder wird ohne `return_aux` übersprungen.

Die primäre Ausgabe bleibt in allen Fällen `[B,2,H,W]` aus `decoder.logits`.
Movable-Logits oder -Wahrscheinlichkeiten werden nicht in MOS-Logits multipliziert
oder nachgelagert als Filter verwendet. Der normale Val08-Punktbenchmark nutzt
die gespeicherte semantische Fusion durch den Modell-Forward selbst.
Die bereits vorhandene explizite `--movable-diagnostics`-Funktion bleibt erhalten
und lädt ebenfalls genau die gespeicherte Modellvariante. Deren zusätzliche
Filterergebnisse sind eine separate Analyse; für die Hauptablation wird der
normale Benchmark **ohne** diesen nachgelagerten Filter benutzt.

### Validierung, Loss, DDP, Resume und Transfer

`semantic_model_settings` in `src/helper/mos_auxiliary.py` validiert die beiden
neuen Optionen zentral. Builder, normaler Trainer, Resume-Prüfung und der
vorhandene Benchmark-Aufbau verwenden diesen Pfad. Unbekannte Werte werden
abgelehnt. `current_bottleneck` verlangt ein binäres
`salsanext_temporal_e2e` mit aktiviertem `mos_auxiliary.task: movable`;
`bottleneck_gate` verlangt zusätzlich `current_bottleneck`. Legacy und Residual
bleiben mit den fehlenden/neuen Legacy-Defaults unverändert nutzbar.

Dataset, Mapping, Masken, `y_movable`, Collate und die bestehende Loss-/DDP-Logik
wurden nicht geändert:

```text
L_Total = L_MOS + mos_auxiliary.loss_weight * L_Movable
```

Beide Aufgaben bleiben gewichtete CE + Lovasz. Der Optimierer wird weiterhin
über alle trainierbaren Parameter gebaut; bestehende Parametergruppen werden
nicht neu strukturiert. Im normalen Training wird `return_aux=True` verwendet,
so dass die angelegten semantischen Parameter am Lossgraphen teilnehmen.

Die tatsächlich aufgelösten Optionen werden unter
`config.run_metadata.model_options` und im Checkpoint zusätzlich unter
`model_options` gespeichert. Die vollständigen neuen Optionen stehen außerdem
explizit in den beiden neuen YAMLs. Fehlende Legacy-Felder werden beim
Resume-Vergleich als `shared_decoder` und `none` normalisiert. Alle anderen
Modellparameter behalten ihren bisherigen exakten Vergleich. Ein Wechsel
zwischen den Ablationen wird als Resume-Fehler abgelehnt.

Striktes Laden bleibt vorgeschrieben. Legacy hat weiterhin ausschließlich die
bisherigen Keys; die neue Architektur ergänzt `semantic_adapter.*`, verwendet
einen strukturierten `movable_head.*` und optional `semantic_gate.*`. Dadurch
kann ein Checkpoint nicht versehentlich mit der anderen Variante geladen werden.

MAE→MOS behält alle bisherigen Format-, Metadaten-, Key-, Shape-, Dtype- und
Finite-Prüfungen für den gemeinsamen Backbone. Nur die ausdrücklich neuen
Semantikmodule und der MOS-Head dürfen frisch bleiben. Das vervollständigte
State-Dict wird weiterhin `strict=True` geladen. Die neuen YAMLs verwenden
für den kontrollierten Vergleich trotzdem ausschließlich Scratch.

### Zwei echte Fullres-Smoke-Jobs vorbereiten

Der vorhandene Launcher `train_mos_p2_2gpu.slurm` wurde gelesen und wiederverwendet:
Partition P2, ein Knoten/eine Slurm-Task, zwei GPUs ohne Typbindung, 16 CPUs,
64 GB Host-RAM, existierender Apptainer-Container und `/hpc`-Bind-Mount,
Arbeitsverzeichnis `/hpc/LiDARSensorForecaster`, statischer Loopback-Rendezvous.
Er startet genau einmal `python3 -m torch.distributed.run` mit zwei Prozessen,
also einem Rank pro GPU. Es wird kein zusätzliches `srun` oder äußeres
`torchrun` ergänzt. Nach Config, Logroot und Gerät werden sämtliche weiteren
Trainer-Argumente unverändert weitergereicht. Launcheränderungen waren nicht nötig.

Für beide Smoke-Jobs gelten dieselben Overrides: `--epochs 1 --max_batches 2`.
Batchgröße bleibt 8 pro GPU, Worker 0, AMP false, Accumulation 1. Zwei Batches
ergeben zwei Optimizer-Updates pro Rank. Der tatsächlich vorhandene
`LMNetWarmup` unterstützt dies: zwei Updates pro Epoche, eine Warmup-Epoche,
also zwei Warmup-Schritte; verwendete LRs rechnerisch 0.005 und 0.01.
Es wird keine reguläre YAML für den Smoke verändert. Die Validierung enthält
bei zwei Batches und zwei Ranks auch Frames jenseits der ersten acht Scans.

**Nur vorbereitete Einreichung; diese Befehle wurden hier nicht ausgeführt:**

```bash
cd /home/koehlbrandt/hpc/LiDARSensorForecaster
mkdir -p /home/koehlbrandt/hpc/logs/slurm

sbatch --parsable --partition=p2 --nodes=1 --ntasks=1 --gres=gpu:2 \
  --chdir=/home/koehlbrandt/hpc/LiDARSensorForecaster \
  --export=ALL,HPC_ROOT=/home/koehlbrandt/hpc \
  --job-name=mos_semanticKitti_temporal_e2e_lmnet_movable_cb_none_smoke \
  --output=/home/koehlbrandt/hpc/logs/slurm/%x-%j.out \
  --error=/home/koehlbrandt/hpc/logs/slurm/%x-%j.err \
  /home/koehlbrandt/hpc/LiDARSensorForecaster/train_mos_p2_2gpu.slurm \
  src/configs/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_none.yaml \
  --epochs 1 --max_batches 2

sbatch --parsable --partition=p2 --nodes=1 --ntasks=1 --gres=gpu:2 \
  --chdir=/home/koehlbrandt/hpc/LiDARSensorForecaster \
  --export=ALL,HPC_ROOT=/home/koehlbrandt/hpc \
  --job-name=mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate_smoke \
  --output=/home/koehlbrandt/hpc/logs/slurm/%x-%j.out \
  --error=/home/koehlbrandt/hpc/logs/slurm/%x-%j.err \
  /home/koehlbrandt/hpc/LiDARSensorForecaster/train_mos_p2_2gpu.slurm \
  src/configs/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate.yaml \
  --epochs 1 --max_batches 2
```

Die Jobs starten unabhängig aus Scratch; es wird weder `--resume` noch
`--pretrained_path` übergeben. Sie dürfen unabhängig voneinander geplant werden;
es wird keine Array-Unterstützung oder zusätzliche Clusterfunktion angenommen.

Slurm-Logs liegen unter `hpc/logs/slurm/<Jobname>-<Job-ID>.out/.err`.
Die getrennten Host-Run- und Checkpointpfade sind:

```text
/home/koehlbrandt/hpc/logs/mos/
  mos_semanticKitti_temporal_e2e_lmnet_movable_cb_none/job-<ID_NONE>/
    MOS_temporal_e2e_lmnet_off8_movable_cb_none_mfmos_w1_<STAMP>/
      config.yaml
      csv/epochs.csv
      checkpoints/{last.pt,best_mean_iou.pt,best_moving_iou.pt}

/home/koehlbrandt/hpc/logs/mos/
  mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate/job-<ID_GATE>/
    MOS_temporal_e2e_lmnet_off8_movable_cb_gate_mfmos_w1_<STAMP>/
      config.yaml
      csv/epochs.csv
      checkpoints/{last.pt,best_mean_iou.pt,best_moving_iou.pt}
```

Im Container entspricht der Präfix `/home/koehlbrandt/hpc` dem Mount `/hpc`.

Status und Exitcode, mit den von `sbatch --parsable` ausgegebenen IDs:

```bash
squeue -u "$USER" -o '%.18i %.9P %.70j %.8T %.10M %R'
sacct -j ID_NONE,ID_GATE --format=JobID,JobName%70,State,ExitCode,Elapsed,AllocTRES%80
```

Abnahme erwartet `COMPLETED` und ExitCode `0:0`. Zusätzlich im jeweiligen
`.out` prüfen: beide Rank-/GPU-Zeilen, `world_size=2`, korrekte `model_options`,
Batch8/global16, erfolgreiche Train- und Val-Batches, endliche MOS-CE/Lovasz und
Movable-CE/Lovasz, `optimizer_updates=2`, `checkpoint_written=.../last.pt`.
Die bisherigen CSV-Felder für Movable-IoU/F1/Precision/Recall, Gesamtparameter,
Peak-VRAM und Laufzeit bleiben erhalten. Die zusätzliche Epochenausgabe enthält
beide Loss-Komponenten und Peak-VRAM für Train und Val.
Die Trainingsvalidierung berechnet Auxiliary mit; die finale MOS-Inferenzmessung
im Point-Benchmark fordert ausschließlich MOS an und berechnet beim Gate nur
Adapter und Modulation zusätzlich.

VRAM und Laufzeit wurden hier nicht gemessen. Beide regulären Konfigurationen
bleiben bei Batch8/GPU. Falls auf dem HPC eine gemeinsame Anpassung nötig wird,
muss sie für beide Ablationen identisch erfolgen; hier wurde keine einseitige
Batch-/AMP-/Workeränderung vorgenommen.

### Gezielte Abnahme auf einem zugewiesenen Compute-Node

Vorbereitet, **nicht ausgeführt**:

```text
tests/test_temporal_mos_semantic_branch.py
tools/check_temporal_mos_semantic_branch.py
```

Die neue Suite enthält 14 vorbereitete Prüfungen für Config-Gleichheit,
Legacy-Defaults/Keys/RNG, ungültige Kombinationen im Builder/Trainer/Benchmark,
identische gemeinsame Initialisierung, Viertelauflösung, einen Encoder- und
Decoder-Forward, Ausgabeformen, `return_aux`/Diagnostik, lazy Movable-Auswertung,
fehlenden MOS-Einfluss ohne Gate, dasselbe S_t für Head und Gate, aktuelle
Semantik vor Attention, Gate-Maske, ersten Gate-Gradienten, MOS-Gradienten nach
S_t bei nichttrivialem P, Movable-Gradienten in beiden Ablationen, striktes
Benchmark-Laden/Roundtrip, Resume-Abgrenzung und MAE-Transfer.

Die optional vorbereitete Zwei-GPU-NCCL-Prüfung macht je zwei synthetische
Optimizer-Updates pro Variante und kontrolliert alle Parametergradienten und
die Optimizer-Abdeckung. Ein Rank erhält beim zweiten Schritt ausschließlich
ignorierte Movable-Targets. Dies benötigt zwei bereits zugewiesene GPUs und
wird nicht aus einem äußeren `torchrun` gestartet.

Im vorhandenen HPC-Container auf dem Compute-Node:

```bash
# Kein dritter vollständiger Trainingslauf: kleine synthetische Strukturchecks
# und striktes Laden des bisherigen Movable-Checkpoints mit der Originalreferenz.
python3 tools/check_temporal_mos_semantic_branch.py \
  --device cuda --run-suite --two-gpu-ddp \
  --legacy-checkpoint /hpc/logs/mos/mos_semanticKitti_temporal_e2e_lmnet_aux_movable/job-6967/MOS_temporal_e2e_lmnet_off8_aux_movable_mfmos_w1_20261031_190312/checkpoints/best_moving_iou.pt
```

Der Legacy-Pfad stammt aus der bereits vorhandenen lokalen Diagnoseanleitung;
falls der Checkpoint anderswo liegt, den tatsächlichen Pfad einsetzen.
`--reference-config` ist standardmäßig die unveränderte lokale Movable-YAML.
Die Strukturchecks verwenden synthetisch 32×64; die beiden echten Trainings-Smokes
verwenden die vollständige Referenzauflösung 64×2048.

Alternativ ohne den realen Legacy-Checkpointcheck:

```bash
MOS_CHECK_DEVICE=cuda MOS_CHECK_DDP=1 python3 -m unittest discover \
  -s tests -p 'test_temporal_mos_semantic_branch.py' -v
```

Ohne `--two-gpu-ddp` bzw. ohne `MOS_CHECK_DDP=1` wird nur der spezielle
Zwei-GPU-Check übersprungen; die übrigen vorbereiteten Checks können auf einem
Compute-Node mit einer GPU laufen. Es wird kein Job automatisch eingereicht.

Nach den beiden Smoke-Jobs, jeweils mit dem tatsächlichen Containerpfad:

```bash
python3 tools/check_temporal_mos_semantic_branch.py --device cuda \
  --checkpoint "$CKPT_NONE_LAST" --expect-fusion none \
  --expect-world-size 2 --expect-min-updates 2
python3 tools/check_temporal_mos_semantic_branch.py --device cuda \
  --checkpoint "$CKPT_GATE_LAST" --expect-fusion bottleneck_gate \
  --expect-world-size 2 --expect-min-updates 2
```

Diese Prüfung lädt den gespeicherten Modus über den echten Benchmark-Adapter
strikt, vergleicht MOS mit/ohne `return_aux`, prüft einen temporären
Checkpoint-Roundtrip und liest die tatsächlichen Train-/Val-Losses und
Optimizer-Updates aus der CSV. Die JSON-Ausgabe enthält Modus, Parameterzahl,
Worldsize und Updatezahl. Die Smoke-Checkpoints sind keine endgültigen
wissenschaftlichen Ergebnisse.

### Reguläre 100-Epochen-Läufe und Val08-Punktbenchmark

Nach erfolgreicher HPC-Abnahme, vom Repository aus mit dem vorhandenen Wrapper:

```bash
bash submit_mos_p2_2gpu.sh \
  src/configs/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_none.yaml
bash submit_mos_p2_2gpu.sh \
  src/configs/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate.yaml
```

Keine Smoke-Overrides, kein Resume, kein MAE-Pretraining. Der Wrapper setzt
P2, zwei GPUs, Repository-Arbeitsverzeichnis und Logpfade. Falls
`P2_GPU_GRES` in der Umgebung bewusst gesetzt wurde, wird es wie bisher
verwendet; ohne diese Variable bleibt `gpu:2` der vorhandene Default.

Nach Abschluss beide `best_mean_iou.pt` separat mit demselben Protokoll auswerten:

```bash
python3 tools/evaluate_mos_benchmark.py \
  --checkpoint "$CKPT_NONE_BEST_MEAN_IOU" \
  --dataset-root /hpc/SemanticKitti/dataset/sequences \
  --sequences 08 --mode point --postprocess knn \
  --knn-k 5 --knn-search 5 --knn-sigma 1.0 --knn-cutoff 1.0 \
  --moving-threshold 0.5 --device cuda \
  --output-dir /hpc/logs/mos/semantic_cb_none_val08_$USER

python3 tools/evaluate_mos_benchmark.py \
  --checkpoint "$CKPT_GATE_BEST_MEAN_IOU" \
  --dataset-root /hpc/SemanticKitti/dataset/sequences \
  --sequences 08 --mode point --postprocess knn \
  --knn-k 5 --knn-search 5 --knn-sigma 1.0 --knn-cutoff 1.0 \
  --moving-threshold 0.5 --device cuda \
  --output-dir /hpc/logs/mos/semantic_cb_gate_val08_$USER
```

Die Ausgabeverzeichnisse müssen neu sein; bestehende Benchmark-Ergebnisse werden
nicht überschrieben. Keine `--max-frames`-Begrenzung für das Endergebnis und kein
`--movable-diagnostics`-Filter im Hauptvergleich. Originalpunkte, Poseverarbeitung,
Mapping und KNN bleiben unverändert. Primärer Vergleich: punktweise Moving-IoU,
Moving-Precision und Moving-Recall der beiden aktuellen kompakten Zweige.

### Wissenschaftliche Abgrenzung und Prüfstand

MF-MOS nutzt einen separaten Semantic Branch und Motion-Guided-Feature-Fusion.
Hier bleibt der gemeinsame Siamese-Encoder mit zeitlicher Bottleneck-Attention
die Basis. Ein kleiner aktueller Semantikadapter plus Skip-Head ersetzt in den
beiden Ablationen den Legacy-Auxiliary-Head; die optionale Rückkopplung ist eine
einzelne begrenzte residuale Bottleneck-Modulation. Das ist eine kompakte
MF-MOS-inspirierte Ablation, keine Architektur-Reproduktion von MF-MOS.

Der Vergleich Legacy gegen den neuen Zweig ändert die semantische Architektur
und ihren direkten Gradientenpfad. **Den isolierten Effekt der Rückkopplung
misst dagegen `current_bottleneck+none` gegen
`current_bottleneck+bottleneck_gate`**, mit identischem Movable-Zweig und
identischen Trainingsbedingungen. Die tatsächlichen Point-Ergebnisse sind noch
offen; es werden weder eine Verbesserung noch ein erfolgreicher Lauf behauptet.

Statisch geprüft: Branch/Refs und lokaler Änderungsstand, vorhandene Dateien und
Aufrufpfade, Encoderkanäle und Skip-Auflösung, Config-Textunterschiede,
RNG-Kontexte, Gradientenpfade, Default-/State-Key-Erhaltung, strikte
Transfer-/Resume-Pfade, gespeicherte Optionen, vorhandener Launcher und
Update-Scheduler sowie `git diff --check`. Zusätzlich wurden die sieben
geänderten/neuen Python-Quelltexte als reine Syntaxbäume geparst, ohne
Projektmodule, PyTorch oder die vorbereiteten Prüfungen auszuführen.

Noch auf dem HPC auszuführen: sämtliche neuen Struktur-/Gradienten-/NCCL-Checks,
das strikte Laden des realen Legacy-Checkpoints, beide Fullres-Smokes,
Checkpoint-Roundtrips, VRAM-/Laufzeitmessung, die regulären Trainings und die
vollständige Val08-Punktauswertung. Alte Testergebnisse aus den vorigen Abschnitten
sind kein Ausführungsnachweis für diese Erweiterung.

Geänderte Kernpfade:
`src/mos_models/salsanext_temporal_e2e.py`, `src/mos_models/builder.py`,
`src/helper/mos_auxiliary.py`, `src/helper/temporal_e2e_transfer.py`,
`tools/train_temporal_mos_e2e.py` und diese bestehende Dokumentation.
Neu sind die beiden lokalen YAMLs, die vorbereitete Testsuite und das separate
Compute-Node-Abnahmewerkzeug. Dataset, Collate, Lossimplementierung,
Projektions-/Augmentationscode, Point-Benchmark und bestehende Launcher wurden
in dieser Erweiterung nicht verändert.
