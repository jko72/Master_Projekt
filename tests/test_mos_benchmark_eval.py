import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from helper.mos_benchmark_eval import (add_counts, point_counts, point_metrics,
    read_point_gt, validate_export, write_prediction, write_submission_zip)
from helper.mos_point_projection import ProjectionConfig, original_points_to_pixels, unproject_prediction
from utils_torch import spherical_projection
from evaluate_mos_benchmark import frame_features, index_sequences, load_adapter, load_poses, main, single_frame_input, temporal_input
from mos_dataset import MOSFrameDataset
from mos_models import build_mos_model


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ProjectionConfig(64, 512, 3, -25)

    def test_matches_active_projection_and_near_wins(self):
        angles = np.array([-np.pi, -2, 0, 2, np.pi], dtype=np.float32)
        pitch = np.deg2rad(np.array([-30, -25, 0, 3, 10], dtype=np.float32))
        radius = np.array([3, 4, 5, 6, 7], dtype=np.float32)
        pts = np.stack([radius * np.cos(pitch) * np.cos(angles),
                        radius * np.cos(pitch) * np.sin(angles),
                        radius * np.sin(pitch), np.ones(5)], axis=1).astype(np.float32)
        pts = np.concatenate([pts, pts[:1] * np.array([[2, 2, 2, 1]], dtype=np.float32)])
        rows, cols, valid = original_points_to_pixels(pts, self.cfg)
        image, _, _, _ = spherical_projection(pts, height=64, width=512,
            theta_range=(np.deg2rad(-25), np.deg2rad(3)))
        self.assertTrue(valid.all())
        self.assertEqual((rows[0], cols[0]), (rows[-1], cols[-1]))
        np.testing.assert_allclose(image[rows[0], cols[0]], pts[0])
        for i in range(1, 5):
            np.testing.assert_allclose(image[rows[i], cols[i]], pts[i])
        pred = np.zeros((64, 512), dtype=np.uint8)
        pred[rows[0], cols[0]] = 1
        labels, _ = unproject_prediction(pred, pts, self.cfg)
        self.assertEqual(labels[0], labels[-1])
        self.assertEqual(len(labels), len(pts))

    def test_invalid_fallback_and_range(self):
        pts = np.array([[1, 0, 0, 1], [np.nan, 0, 0, 1],
                        [np.inf, 0, 0, 1], [0, 0, 0, 1], [100, 0, 0, 1]], np.float32)
        cfg = ProjectionConfig(64, 512, 3, -25, max_range=80)
        labels, valid = unproject_prediction(np.ones((64, 512), np.uint8), pts, cfg)
        np.testing.assert_array_equal(valid, [True, False, False, False, False])
        np.testing.assert_array_equal(labels, [1, 0, 0, 0, 0])

    def test_single_frame_features_equal_existing_dataset(self):
        pts = np.array([[2, 0, 0, 1], [3, 1, -0.2, 0.5], [4, -1, 0.1, 0.2]], np.float32)
        cfg = {"model_params": {"grid_height": 64, "grid_width": 512, "FOV_UP": 3, "FOV_DOWN": -25}}
        with tempfile.TemporaryDirectory() as folder:
            scan = Path(folder) / "000000.bin"
            pts.tofile(scan)
            original = MOSFrameDataset.__new__(MOSFrameDataset)
            original.H, original.W = 64, 512
            original.theta_range = [np.deg2rad(-25), np.deg2rad(3)]
            existing = original._load_projected_xyzd(str(scan))
            fresh = frame_features(pts, cfg, {"input_mode": "range_xyz"})
            for a, b in zip([existing[1], existing[2], existing[3], existing[0]], fresh):
                np.testing.assert_array_equal(a, b)

    def test_projected_winner_ids_match_vectorized_coordinates(self):
        rng = np.random.default_rng(31)
        points = rng.normal(size=(8000, 4)).astype(np.float32)
        points[:, :3] *= rng.uniform(1, 70, size=(8000, 1)).astype(np.float32)
        points[:, 3] = np.arange(1, len(points) + 1, dtype=np.float32)
        rows, cols, valid = original_points_to_pixels(points, self.cfg)
        image, _, _, _ = spherical_projection(points, height=64, width=512,
            theta_range=(np.deg2rad(-25), np.deg2rad(3)))
        winners = image[..., 3].astype(np.int64)
        occupied = winners > 0
        selected = winners[occupied] - 1
        actual_rows, actual_cols = np.where(occupied)
        self.assertTrue(valid[selected].all())
        np.testing.assert_array_equal(rows[selected], actual_rows)
        np.testing.assert_array_equal(cols[selected], actual_cols)


class BenchmarkTests(unittest.TestCase):
    def test_gt_mapping_and_global_iou(self):
        with tempfile.TemporaryDirectory() as folder:
            gt = Path(folder) / "000000.label"
            np.array([0, 1, 9, 10, 99, 251, 259, 777,
                      (42 << 16) | 252], dtype="<u4").tofile(gt)
            mapped = read_point_gt(gt, 9)
            np.testing.assert_array_equal(mapped, [-1, -1, 0, 0, 0, 1, 1, -1, 1])
            with self.assertRaises(ValueError):
                read_point_gt(gt, 8)
        total = {}
        add_counts(total, point_counts(np.array([1, 1]), np.array([1, 0])))
        add_counts(total, point_counts(np.array([0, 0, 0, 0]), np.array([1, 0, 0, -1])))
        self.assertAlmostEqual(point_metrics(total)["point_moving_iou"], 1 / 3)
        self.assertEqual(total["n_gt_valid"], 5)

    def test_export_and_label_free_index(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scan_dir = root / "dataset" / "sequences" / "11" / "velodyne"
            scan_dir.mkdir(parents=True)
            scan = scan_dir / "000000.bin"
            np.array([[1, 0, 0, 1], [2, 0, 0, 1]], np.float32).tofile(scan)
            indexed = index_sequences(scan_dir.parent.parent, ["11"])
            out = root / "out" / "predictions"
            label = out / "sequences" / "11" / "predictions" / "000000.label"
            write_prediction(label, np.array([0, 1], np.uint8))
            validate_export(out, indexed)
            self.assertEqual(label.stat().st_size, 8)
            np.testing.assert_array_equal(np.fromfile(label, dtype="<u4"), [9, 251])
            with self.assertRaises(ValueError):
                write_submission_zip(root / "submission.zip", out, indexed)

    def test_adapter_contract_and_temporal_bootstrap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bad = root / "bad.pt"
            torch.save({"cfg": {"mos_model_params": {"name": "salsanext_mos"}}}, bad)
            with self.assertRaisesRegex(ValueError, "model_state_dict"):
                load_adapter(bad, torch.device("cpu"))
            scan = root / "000000.bin"
            np.array([[1, 0, 0, 1]], np.float32).tofile(scan)
            cfg = {"mos_data_params": {"previous_offsets": [2, 1], "scan_rate_hz": 10},
                   "model_params": {"grid_height": 8, "grid_width": 16, "FOV_UP": 3, "FOV_DOWN": -25}}
            args, boot = temporal_input([scan], 0, [np.eye(4)], cfg, "repeat_earliest", True)
            self.assertTrue(boot)
            self.assertEqual(tuple(args[0].shape), (1, 3, 4, 8, 16))
            np.testing.assert_allclose(args[2].numpy(), [[-.2, -.1, 0]])
            self.assertFalse(args[1][0, :-1].any())
            with self.assertRaisesRegex(ValueError, "Missing history"):
                temporal_input([scan], 0, [np.eye(4)], cfg, "strict", True)
            with self.assertRaisesRegex(FileNotFoundError, "poses.txt AND calib.txt"):
                load_poses(root, 1)

    def test_label_free_single_frame_inference_and_residual_strictness(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scan_dir = root / "dataset" / "sequences" / "11" / "velodyne"
            scan_dir.mkdir(parents=True)
            scan = scan_dir / "000000.bin"
            np.array([[2, 0, 0, 1], [3, 1, 0, 1]], np.float32).tofile(scan)
            cfg = {"model_params": {"grid_height": 16, "grid_width": 32, "FOV_UP": 3, "FOV_DOWN": -25},
                   "mos_model_params": {"name": "unet_small", "in_channels": 4, "num_classes": 2,
                                        "base_channels": 8},
                   "mos_data_params": {"input_mode": "range_xyz", "train_sequences": ["00"]}}
            ckpt = root / "model.pt"
            torch.save({"cfg": cfg, "model_state_dict": build_mos_model(cfg).state_dict()}, ckpt)
            out = root / "eval"
            main(["--checkpoint", str(ckpt), "--dataset-root", str(root / "dataset"),
                  "--sequences", "11", "--mode", "point", "--export-predictions",
                  "--output-dir", str(out), "--device", "cpu"])
            pred = out / "predictions/sequences/11/predictions/000000.label"
            self.assertTrue(pred.is_file())
            self.assertEqual(len(np.fromfile(pred, dtype="<u4")), 2)
            self.assertFalse((out / "point_metrics.csv").exists())
            cfg["mos_data_params"] = {"input_mode": "range_residual", "residual_offsets": [1]}
            with self.assertRaisesRegex(FileNotFoundError, "residual image missing"):
                single_frame_input(scan, cfg, scan_dir.parent, scan.stem)


if __name__ == "__main__":
    unittest.main()
