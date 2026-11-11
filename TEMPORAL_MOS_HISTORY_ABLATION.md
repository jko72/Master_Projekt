# History-/Temporal-Ablation der bestehenden MOS-Pipeline

## Fragestellung und Grenzen

Die Inferenzintervention untersucht für dasselbe trainierte Netzwerk, wie sich
fehlende historische LiDAR-Messungen und das vollständige Umgehen der temporalen
Cross-Attention auf MOS auswirken. Es ist kein neues Training erforderlich:
Alle Modi laden denselben bestehenden Checkpoint mit `strict=True`, ohne Gewichte
zu verändern. Architekturparameter, State-Dict-Keys und Parameterzahl bleiben gleich.

Diese Ablationen ersetzen **keine fair trainierte Single-Frame-Baseline**.
`mask_history` verwendet weiterhin gelernte No-Match-Parameter und den aktuellen
Attention-Query-/Vergleichspfad. `bypass_temporal` entfernt diesen gesamten
Berechnungsschritt und verändert dadurch die Eingabeverteilung des nachfolgenden
Gates und Decoders. Beide Verfahren sind diagnostische Inferenzinterventionen.

## Drei Modi

| CLI `--temporal-ablation` | Historische Inputs | Cross-Attention | Semantic Gate |
|---|---|---|---|
| `none` (Default) | unverändert | unverändert aktiv | unverändert aktiv, falls konfiguriert |
| `mask_history` | alle historischen Features null, alle historischen Valid-Masken falsch | aktiv, historische Kandidaten ungültig, No-Match-Token gültig | unverändert aktiv |
| `bypass_temporal` | normal vorbereitet | vollständig übersprungen, `fused = encoded[:, -1]` | unverändert auf dem aktuellen Bottleneck angewendet |

`mask_history` klont `x` und `valid` unmittelbar vor dem Forward. Aktueller Scan,
aktuelle Maske, `dt`, Tensorformen, Ground Truth und Punktreihenfolge bleiben
unverändert. Zuvor läuft der normale `temporal_input()`-Pfad einschließlich
Offset-Auswahl aus `previous_offsets`, Pose und Egomotion-Kompensation.
Das vorhandene No-Match-Token in `LocalHistoricalAttention` wird nicht verändert.

`SalsaNextTemporalE2E.forward()` erhält als letzten optionalen Parameter
`disable_temporal_fusion=False`. Der Default folgt exakt dem bisherigen Pfad.
Bei `True` wird `self.temporal(...)` nicht aufgerufen; danach folgen unverändert
Gate-Fusion, Decoder und aktuelle Skip-Connections. Der bestehende gemeinsame
Encoder-Forward wird beibehalten; es wird keine zusätzliche Encoder-Optimierung
eingeführt. `return_aux` funktioniert weiterhin. Die Kombination mit
`return_diagnostics=True` führt zu einem verständlichen `ValueError`, weil bei
umgangener Attention keine Attention-Diagnostik vorliegt.

Der Modellparameter `disable_semantic_gate` bleibt technisch kompatibel. Im
Benchmark wird seine Kombination mit einer aktiven Temporal-Ablation jedoch
absichtlich abgelehnt, damit getrennte Forschungsfragen nicht vermischt werden.
Aktive Temporal-Ablationen erlauben ausschließlich Threshold 0,5 ohne
Threshold-Sweep oder zusätzliches Movable-Diagnostik-Postfiltering.
Der bisherige Default einschließlich Threshold-Sweep/Gate-Ablation bleibt erhalten.
Andere Modelltypen werden für die neuen aktiven Modi abgelehnt.

## Unverändertes Evaluationsprotokoll

Für beide Checkpoints werden jeweils drei neue, getrennte Evaluationen ausgeführt:
SemanticKITTI Validation 08, 4.071 Scans, 64×2048, XYZRI, LMNet-Projektion,
Referenzoffset 8, Moving-Threshold 0,5 und KNN `k=5`, `search=5`, `sigma=1.0`,
`cutoff=1.0`, Chunkgröße 32768. KNN-Code, Losses, Training, Dataloader und
Trainingsconfigs werden nicht geändert. Die Trainingssampling-Sektion des zweiten
Checkpoints wird vom bestehenden Benchmark-Input-Builder weiterhin ignoriert.

| Checkpoint | Epoche geprüft | bisheriges Referenzergebnis Point-Moving-IoU |
|---|---|---|
| cb_gate, Training mit festem Offset 8, Job 8228 | 85 | ca. 61,1347 % |
| cb_gate mit Temporal-Sampling, Job 8307 | 69 | ca. 58,6876 % |

Beide angegebenen Dateien existieren im aktuellen HPC-Dateisystem. Ihre SHA-256:

- Fest: `ff7b3a2cb5c4477262dc243d10d58c2c6fecce88ae21ae1ff2a594a09603cfd1`
- Sampling: `bd089e1da14491a962674c333542069640943d5c01a7b08825db0b094dc26e06`

Die historischen Referenzresultate sind Vergleichswerte; die neuen vollständigen
Evaluationen müssen diese zunächst im Modus `none` reproduzieren.

## Metadaten und Vergleichbarkeit

`benchmark_eval_config.yaml` enthält zusätzlich:

```yaml
temporal_ablation:
  mode: mask_history
  checkpoint_unchanged: true
  historical_inputs_masked: true
  temporal_module_executed: true
  semantic_gate_unchanged: true
history_reference_offsets: [8]
```

`none` setzt `historical_inputs_masked:false`, `temporal_module_executed:true`.
`bypass_temporal` setzt beide auf `false`. Für eine separate Gate-Ablation bei
`none` wird `semantic_gate_unchanged:false` dokumentiert. Bei nicht unterstützten
Modelltypen im bisherigen Defaultpfad ist `temporal_module_executed:false`
auf das hier untersuchte `LocalHistoricalAttention`-Modul bezogen.
Die bestehende `semantic_gate_ablation`-Sektion und alle übrigen Metadaten bleiben
erhalten: Checkpoint-Pfad, Hash, Epoche, vollständige Checkpointconfig, Git-Commit,
Sequenzen, Projektion, Normalisierung, Postprocessing, Threshold und Scananzahl.

In `projection_coverage.csv` beschreibt `history_available` weiterhin die
ursprüngliche Verfügbarkeit. Die zusätzliche Spalte `history_used` beschreibt
die effektive Nutzung als gültige historische Attention-Kandidaten. Sie ist bei
`mask_history` und `bypass_temporal` immer falsch, auch wenn reale Scans verfügbar
sind. Die bestehende Bedeutung von `history_available` wird nicht überschrieben.

`tools/compare_temporal_ablation.py` validiert für alle drei Modi:

- identischen Checkpoint-Pfad/-Hash/-Epoche und vollständige Modellconfig;
- identische Sequenzen, Scan-IDs, Reihenfolge, Punktzahlen, physische Historie;
- identischen Referenzoffset, Threshold, KNN, Projektion und Normalisierung;
- identische Datenwurzel, Pose-/GT-Protokolle, Smoke-Kennzeichnung und Git-Commit;
- passende Temporal-Modi, unveränderten Checkpoint und unveränderte Gate-Aktivierung;
- keine zusätzlichen Gate-Ablationen, Sweeps oder Movable-Diagnostik;
- gültige nichtnegative ganzzahlige Confusion-Zähler und konsistente Punkt-/Scan-
  sowie Ground-Truth-Klassensummen.

Die vier Moving-Metriken IoU, Precision, Recall und F1 werden aus TP/FP/FN/TN
neu berechnet. Gespeicherte IoU-Werte werden nicht als Berechnungsgrundlage verwendet.
Pixel-Metriken werden ebenfalls verglichen, wenn sie für alle drei Modi vorliegen.
Bei nur teilweise vorhandenen Pixel-Metriken wird der Vergleich abgelehnt.
Differenzen sind **mask minus normal** beziehungsweise **bypass minus normal**,
bei den vier Raten zusätzlich in **Prozentpunkten**.

Ausgaben: `temporal_ablation_comparison.csv`, `temporal_ablation_comparison.yaml`
und eine Konsolentabelle. Bestehende Output-Verzeichnisse werden nicht überschrieben.

## Ergänzende Gamma-Diagnostik

Die rein gelesene `temporal_attention_gamma`-Sektion enthält `gamma_mean`,
`gamma_mean_abs`, `gamma_min`, `gamma_max` aus dem strikt geladenen Checkpoint.
Für Epoche 85 beträgt der mittlere Absolutwert ca. 0,00360346, für Epoche 69
ca. 0,00400007. Ein kleiner Skalierungswert allein beweist keinen kleinen
Einfluss: Entscheidend sind auch die skalierten Features und nachfolgenden
Berechnungen. Diese Diagnose ersetzt die Ablation nicht.

## Tests und Checkpoint-Kompatibilität

Neue Tests: `tests/test_temporal_history_ablation.py` (acht Tests).
Sie prüfen Default/`none` einschließlich RNG- und State-Unveränderlichkeit,
Kopien und Maskierung aller Historien, No-Match-Gewicht exakt eins,
finite Logits, vollständiges Überspringen der Attention, exaktes aktuelles
Bottleneck vor dem Gate, unveränderte Gate-Formel und aktuelle Skips,
Auxiliary-Outputs, Diagnostik-Fehler, State-Dict-/Parameterkompatibilität,
synthetische vollständige KNN-Point-/Pixel-Benchmarks mit echter Offset-8-Historie,
unveränderte GT-/Checkpoint-Hashes, Metadaten sowie Vergleich und Ablehnung
inkompatibler Protokolle. Absichtlich verfälschte gespeicherte IoUs prüfen,
dass der Vergleich tatsächlich aus Confusion-Zählern rechnet.

Beide echten Checkpoints wurden zusätzlich lokal strikt geladen. Ein separat
gesicherter Modellcode vor dieser Änderung liefert mit denselben Gewichten
bitweise identische Default-Outputs auf CPU. Alle drei Modi liefern finite
Outputs. Beide Modelle behalten **7.144.964 Parameter**, dieselben State-Dict-Keys
und unveränderte Checkpoint-Hashes.

```bash
MPLCONFIGDIR=/tmp/mos-mpl OMP_NUM_THREADS=2 /tmp/mos-lovasz-venv/bin/python \
  -m unittest discover -s tests -p 'test_temporal_history_ablation.py' -v

PYTHONPATH=tests MPLCONFIGDIR=/tmp/mos-mpl OMP_NUM_THREADS=2 \
  /tmp/mos-lovasz-venv/bin/python -m unittest \
  test_semantic_gate_eval_ablation test_mos_benchmark_eval -v
```

Die acht neuen Tests und alle 27 bestehenden Gate-/Benchmarktests bestehen.
Syntax-/Importprüfung, Bash-Syntax der Launcher und `git diff --check` wurden
zusätzlich geprüft. Die breitere Temporal-Regression wurde in einer temporären
Kopie mit den vorhandenen Archivconfigs an ihren alten Testpfaden ausgeführt; die tatsächlichen
Trainingsconfigs werden dafür nicht verändert. Die bereits zuvor gegen den
unveränderten HEAD nachgewiesenen zwei Residual-Referenzfehler werden separat
ausgewiesen. Lokale CPU-Umgebung: PyTorch 2.14.1, Python 3.12; HPC-Container:
PyTorch 2.5.1+cu121, Python 3.10.

Ergebnis der breiteren Regression: 81 Tests, davon 77 bestanden, zwei übersprungen
und zwei bestehende Residual-Fehler (`test_config_and_protocol` und
`test_distributed_accumulation_reference`). Die acht neuen Tests sind in diesen
81 Tests enthalten. Zusammen mit den 27 Gate-/Benchmarktests sind es 108
unterschiedliche Tests: 104 bestanden, zwei übersprungen, zwei bekannte Fehler.
Die gesamte Altsuite wird daher ausdrücklich nicht als vollständig grün bezeichnet.
Diese Fehler wurden bereits bei der Offset-Sampling-Implementierung gegen
unveränderten Code reproduziert und gehören nicht zur Inferenzintervention.

Prüflogs, Prüfbericht beider echter Checkpoints, Source-Snapshot, Git-Diff und
SHA-256-Manifest liegen dauerhaft unter
`/hpc/logs/mos/evaluations/temporal_ablation_verification_20261111`.

## HPC-Start und Ergebnisordner

Der neue Launcher übernimmt die Ressourcen des vorhandenen
`evaluate_semantic_gate_ablation_p1.slurm` unverändert: P1, ein Node, eine GPU,
acht CPUs, 32 GB RAM, zwölf Stunden. Es wird dasselbe bestehende Apptainer-Image
mit lokaler SIF-Kopie verwendet. Ein Job bearbeitet einen Checkpoint und führt
alle drei Modi sequenziell aus. Zuerst laufen zehn Smoke-Scans pro Modus,
einschließlich tatsächlich verfügbarer Offset-8-Historie. Bei einem Smoke-Fehler
stoppt das Skript vor dem vollständigen Benchmark. Danach folgen die drei
vollständigen Läufe und der automatische Vergleich.

Der Nutzer hat mit der aktuellen Nachricht auch den Start auf P1 autorisiert.
Daher werden die beiden vollständigen Jobs nach den lokalen Prüfungen gestartet;
die ältere Vorgabe im Anhang, nur vorzubereiten, ist dadurch aktualisiert.

Auf dem HPC-Login-Host:

```bash
cd /home_cu/s210891/LiDARSensorForecaster

bash submit_temporal_ablation_p1.sh \
  /home_cu/s210891/logs/mos/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate/job-8228/MOS_temporal_e2e_lmnet_off8_movable_cb_gate_mfmos_w1_20261104_154338/checkpoints/best_moving_iou.pt

bash submit_temporal_ablation_p1.sh \
  /home_cu/s210891/logs/mos/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate_temporal_sampling/job-8307/MOS_temporal_e2e_lmnet_cb_gate_temporal_sampling_5_8_10_w1_20261109_135024/checkpoints/best_moving_iou.pt
```

Für ausschließlich Smoke-Tests kann jeweils `--smoke-only` angehängt werden.
Die gestarteten Job-IDs sind **8361 (fester Offset)** und **8362 (Sampling)**.
Beide P1-Smoke-Triplets einschließlich automatischem Point-/Pixel-Vergleich
sind erfolgreich abgeschlossen. Jeweils zehn Scans, acht ohne reale Historie;
bei den letzten beiden Scans bleibt `history_available=[True]`, während
`history_used` nur im normalen Modus `[True]` und in beiden Interventionen
`[False]` ist. Alle fünf Temporal-Metadatenfelder stimmen mit dem jeweiligen
Modus überein, und die Checkpoint-Hashes entsprechen den vorab geprüften Dateien.
Beide Jobs haben anschließend den vollständigen Lauf `full/normal` gestartet.
Ihre Container-Output-Wurzeln:

- `/hpc/logs/mos/evaluations/temporal_ablation_job-8361`
- `/hpc/logs/mos/evaluations/temporal_ablation_job-8362`

Darunter jeweils `smoke/normal`, `smoke/mask_history`, `smoke/bypass_temporal`,
`smoke/comparison` sowie dieselben vier Ordner unter `full/`.
Auf dem Host entspricht `/hpc` dem Verzeichnis `/home_cu/s210891`, lokal
`/home/koehlbrandt/hpc`.

Die drei Einzelbefehle können innerhalb eines bereits zugewiesenen GPU-Containers
auch mit einer frischen gemeinsamen Output-Wurzel ausgeführt werden:

```bash
CHECKPOINT=/hpc/logs/mos/mos_semanticKitti_temporal_e2e_lmnet_movable_cb_gate/job-8228/MOS_temporal_e2e_lmnet_off8_movable_cb_gate_mfmos_w1_20261104_154338/checkpoints/best_moving_iou.pt
RESULT_ROOT=/hpc/logs/mos/evaluations/temporal_ablation_manual_fresh
for ABLATION in none mask_history bypass_temporal; do
  NAME="$ABLATION"
  if [[ "$ABLATION" == none ]]; then NAME=normal; fi
  python3 -u tools/evaluate_mos_benchmark.py \
    --checkpoint "$CHECKPOINT" \
    --dataset-root /hpc/SemanticKitti/dataset/sequences \
    --sequences 08 --mode both --device cuda \
    --postprocess knn --knn-k 5 --knn-search 5 \
    --knn-sigma 1.0 --knn-cutoff 1.0 --knn-chunk-size 32768 \
    --moving-threshold 0.5 --temporal-ablation "$ABLATION" \
    --output-dir "$RESULT_ROOT/$NAME"
done
```

Vergleich eines fertigen Triplets (der Launcher erstellt `comparison` bereits
automatisch, deshalb für einen zusätzlichen manuellen Vergleich ein frischer Name):

```bash
EVAL_ROOT=/hpc/logs/mos/evaluations/temporal_ablation_job-8361/full
python3 tools/compare_temporal_ablation.py \
  --normal-dir "$EVAL_ROOT/normal" \
  --mask-history-dir "$EVAL_ROOT/mask_history" \
  --bypass-temporal-dir "$EVAL_ROOT/bypass_temporal" \
  --output-dir "$EVAL_ROOT/comparison_manual"
```

Für den Sampling-Checkpoint `8361` durch `8362` ersetzen. Bei Ausführung außerhalb
des Containers müssen die Pfade zur jeweiligen Host-/SSHFS-Wurzel passen.
Status und Logs auf dem Login-Host:

```bash
squeue -j 8361,8362
tail -f /home_cu/s210891/logs/slurm/mos_temporal_ablation-8361.out
```

## Interpretation

Eine negative Differenz bei `mask_history` zeigt eine Verschlechterung dieses
Checkpoints ohne historische Messungen, während seine gelernte Attention samt
No-Match-Pfad erhalten bleibt. Die Differenz bei `bypass_temporal` umfasst
zusätzlich den Entfall dieses Pfads. Ist Maskieren nahezu folgenlos, aber Bypass
schädlich, kann der aktuelle Query-/No-Match-Pfad eine Rolle spielen. Verbesserungen
unter einer Intervention können auf ungünstige temporale Einflüsse oder
Verteilungsänderungen hinweisen; daraus folgt keine Aussage über die Qualität
einer separat trainierten Single-Frame-Architektur. Diese Aussagen gelten jeweils
für die untersuchten Checkpoints, Daten und den festgelegten Betriebspunkt.

## Geänderte und neue Dateien

- Geändert: `src/mos_models/salsanext_temporal_e2e.py`
- Geändert: `tools/evaluate_mos_benchmark.py`
- Neu: `tools/compare_temporal_ablation.py`
- Neu: `tests/test_temporal_history_ablation.py`
- Neu: `scripts/eval_temporal_ablation_hpc.sh`
- Neu, HPC-lokal: `evaluate_temporal_ablation_p1.slurm`
- Neu, HPC-lokal: `submit_temporal_ablation_p1.sh`
- Neu: `TEMPORAL_MOS_HISTORY_ABLATION.md`

Die vorhandene `.git/info/exclude` hält Slurm-/Submit-Dateien weiterhin lokal.
Bestehende Gate-Ablationsdateien und der Code von `LocalHistoricalAttention`
wurden nicht verändert. Trainingsdateien, Dataloader, Losses, KNN und Configs
bleiben unverändert. Die vollständigen P1-Ergebnisse stehen bei Abschluss der
Implementierung noch aus; die Jobs laufen weiter und schreiben ihre Vergleiche
nach Abschluss aller drei vollständigen Modi automatisch.
