"""Point MOS labels, confusion counts, and official prediction validation."""

from pathlib import Path
import zipfile

import numpy as np

from helper.mos_label_helper import map_semantickitti_to_mos

TEST_SEQUENCES = tuple(f"{i:02d}" for i in range(11, 22))


def read_scan(path):
    raw = np.fromfile(path, dtype="<f4")
    if raw.size % 4:
        raise ValueError(f"Invalid scan float count: {path}")
    return raw.reshape(-1, 4)


def read_point_gt(path, n_points):
    raw = np.fromfile(path, dtype="<u4")
    if len(raw) != n_points:
        raise ValueError(f"GT/scan point count mismatch: {path}: {len(raw)} != {n_points}")
    return map_semantickitti_to_mos(raw & np.uint32(0xFFFF))


def point_counts(pred, target):
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.shape != target.shape or pred.ndim != 1:
        raise ValueError("Point prediction/GT length mismatch")
    if not np.isin(pred, [0, 1]).all() or not np.isin(target, [-1, 0, 1]).all():
        raise ValueError("Invalid binary point labels")
    valid = target != -1
    return {"tp_moving": int(np.sum((pred == 1) & (target == 1) & valid)),
            "fp_moving": int(np.sum((pred == 1) & (target == 0) & valid)),
            "fn_moving": int(np.sum((pred == 0) & (target == 1) & valid)),
            "tn_static": int(np.sum((pred == 0) & (target == 0) & valid)),
            "n_gt_valid": int(valid.sum())}


def add_counts(total, current):
    for key in current:
        total[key] = total.get(key, 0) + int(current[key])
    return total


def point_metrics(counts):
    tp, fp, fn, tn = (counts[k] for k in ("tp_moving", "fp_moving", "fn_moving", "tn_static"))
    ratio = lambda numerator, denominator: numerator / denominator if denominator else 0.0
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    miou, siou = ratio(tp, tp + fp + fn), ratio(tn, tn + fp + fn)
    return {"point_moving_iou": miou, "point_moving_precision": precision,
            "point_moving_recall": recall,
            "point_moving_f1": ratio(2 * precision * recall, precision + recall),
            "point_static_iou": siou, "point_miou": (miou + siou) / 2}


def write_prediction(path, point_pred):
    pred = np.asarray(point_pred)
    if pred.ndim != 1 or not np.isin(pred, [0, 1]).all():
        raise ValueError("Prediction must be a one-dimensional binary array")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    np.where(pred == 1, 251, 9).astype("<u4").tofile(path)
    validate_prediction(path, len(pred))


def validate_prediction(path, n_points):
    path = Path(path)
    if path.stat().st_size != n_points * 4:
        raise ValueError(f"Wrong prediction byte count: {path}")
    raw = np.fromfile(path, dtype="<u4")
    if not np.isin(raw, [9, 251]).all():
        raise ValueError(f"Invalid prediction IDs: {path}")


def validate_export(prediction_root, sequences):
    root = Path(prediction_root)
    for seq_id, scans in sequences.items():
        folder = root / "sequences" / seq_id / "predictions"
        expected = {scan.stem: scan for scan in scans}
        actual = {path.stem: path for path in folder.glob("*.label")}
        if actual.keys() != expected.keys():
            raise ValueError(f"Incomplete prediction export for {seq_id}: expected {len(expected)}, got {len(actual)}")
        for stem, scan in expected.items():
            validate_prediction(actual[stem], scan.stat().st_size // 16)


def write_submission_zip(path, prediction_root, sequences):
    if set(sequences) != set(TEST_SEQUENCES):
        raise ValueError("Submission ZIP requires every official test sequence 11..21")
    validate_export(prediction_root, sequences)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("sequences/", "")
        for seq_id in TEST_SEQUENCES:
            archive.writestr(f"sequences/{seq_id}/", "")
            archive.writestr(f"sequences/{seq_id}/predictions/", "")
            for file in sorted((Path(prediction_root) / "sequences" / seq_id / "predictions").glob("*.label")):
                archive.write(file, f"sequences/{seq_id}/predictions/{file.name}")
