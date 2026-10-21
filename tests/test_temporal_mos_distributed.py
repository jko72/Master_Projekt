"""Two-process CPU/Gloo regression checks; no real dataset or GPU is used."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import train_temporal_mos_e2e as trainer


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, sequences=None, cfg=None, split=None):
        self.size = cfg.get("test_samples", 5)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        x = torch.randn(2, 5, 2, 3, generator=generator)
        valid = torch.ones(2, 2, 3, dtype=torch.bool)
        dt = torch.tensor([-0.5, 0.0])
        y = torch.full((2, 3), 0 if index % 3 == 0 else 1, dtype=torch.long)
        y[0, 0] = -1
        return x, valid, dt, y, {"seq_id": "00", "frame_stem": str(index)}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 2, 1)
        self.register_buffer("marker", torch.zeros(1))

    def forward(self, x, valid, dt):
        return self.conv(x[:, -1])


def batch(dataset, indices):
    items = [dataset[index] for index in indices]
    return [torch.stack([item[column] for item in items]) for column in range(4)]


def distributed_worker(rank, directory):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group("gloo", init_method=f"file://{directory / 'rendezvous'}", rank=rank, world_size=2)
    try:
        device = torch.device("cpu")
        _, cfg = trainer.read_config(ROOT / "src/configs/mos_semanticKitti_temporal_e2e_intensity.yaml")
        cfg["test_samples"] = 5
        with patch.object(trainer, "make_sequences", return_value=[]), \
             patch.object(trainer, "select_sequences", return_value=[]), \
             patch.object(trainer, "TemporalMOSE2EDataset", TinyDataset):
            loader = trainer.make_loader(cfg, "train", 3, 0, device)
            model = trainer.wrap_distributed(TinyModel(), device)
            reference = TinyModel()
            reference.load_state_dict(trainer.unwrap_model(model).state_dict())
            criterion = trainer.make_criterion(cfg, device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
            scaler = torch.amp.GradScaler("cuda", enabled=False)

            indices = [None, None]
            dist.all_gather_object(indices, list(loader.sampler))
            x, valid, dt, y = batch(loader.dataset, indices[0] + indices[1])
            reference_loss = criterion(reference(x, valid, dt), y)
            reference_loss.backward()
            reference_optimizer.step()
            train_result = trainer.run_phase(model, loader, criterion, device, optimizer, scaler)
            assert train_result["samples"] == 6  # DistributedSampler pads only training.
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

            # Validation shards have different lengths and must not pad or deadlock.
            val_loader = trainer.make_loader(cfg, "val", 2, 0, device)
            trainer.unwrap_model(model).marker.fill_(rank)
            val_result = trainer.run_phase(model, val_loader, criterion, device)
            assert trainer.unwrap_model(model).marker.item() == 0
            x, valid, dt, y = batch(TinyDataset(cfg=cfg), range(5))
            with torch.no_grad():
                logits = reference(x, valid, dt)
                expected_loss = criterion(logits, y).item()
            assert val_result["samples"] == 5
            assert abs(val_result["loss"] - expected_loss) < 1e-6
            expected_metrics = trainer.metrics_from_counts(trainer.compute_counts_from_pred_target(logits.argmax(1), y))
            for key, value in expected_metrics.items():
                assert abs(val_result[key] - value) < 1e-7

            # A rank with an empty validation shard must still join the reductions.
            cfg["test_samples"] = 1
            tiny_val = trainer.make_loader(cfg, "val", 2, 0, device)
            assert trainer.run_phase(model, tiny_val, criterion, device)["samples"] == 1

            # Exercise accumulation (including a short final group), two epochs,
            # CLI overrides, single-writer output, and reloadable raw checkpoints.
            cfg["test_samples"] = 9
            cfg["mos_train_params"].update(epochs=7, batch_size=8, num_workers=0, gradient_accumulation=2)
            cfg["mos_log_params"].update(log_root=str(directory / "runs"), use_tensorboard=False)
            config_path = directory / "config.yaml"
            if rank == 0:
                config_path.write_text(yaml.safe_dump(cfg))
            dist.barrier()
            args = SimpleNamespace(cfg_path=str(config_path), log_root=None, device="cpu", dry_run=False,
                                   epochs=2, batch_size=2, num_workers=0, max_batches=None)
            with patch.object(trainer, "build_mos_model", side_effect=lambda _cfg: TinyModel()):
                trainer.train(args, device)
            dist.barrier()
            if rank == 0:
                runs = list((directory / "runs").iterdir())
                assert len(runs) == 1
                run = runs[0]
                checkpoint = torch.load(run / "checkpoints/last.pt", weights_only=False)
                assert checkpoint["epoch"] == 2
                TinyModel().load_state_dict(checkpoint["model_state_dict"], strict=True)
                resolved = yaml.safe_load((run / "config.yaml").read_text())
                assert resolved["mos_train_params"]["batch_size"] == 2
                assert resolved["mos_train_params"]["epochs"] == 2
                assert resolved["run_metadata"]["effective_global_batch_size"] == 8
                with (run / "csv/epochs.csv").open() as handle:
                    assert len(list(csv.DictReader(handle))) == 2
                assert (run / "checkpoints/best_moving_iou.pt").is_file()

        # The real temporal attention model must participate in two DDP backwards
        # without unused-parameter/reducer failures (both temporal input lengths).
        real_model = trainer.wrap_distributed(trainer.build_mos_model(cfg).to(device), device)
        for frames in (2, 3):
            real_model.zero_grad(set_to_none=True)
            x = torch.randn(1, frames, 5, 16, 32)
            valid = torch.ones(1, frames, 16, 32, dtype=torch.bool)
            dt = torch.tensor([[-0.1 * offset for offset in range(frames - 1, -1, -1)]])
            y = torch.randint(0, 2, (1, 16, 32))
            loss = criterion(real_model(x, valid, dt), y)
            loss.backward()
            assert torch.isfinite(loss)
            for parameter in real_model.parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                rank_zero_gradient = parameter.grad.clone()
                dist.broadcast(rank_zero_gradient, src=0)
                torch.testing.assert_close(parameter.grad, rank_zero_gradient)

        overflow_calls = []

        def overflow_once(gradient):
            overflow_calls.append(1)
            return torch.full_like(gradient, float("inf")) if len(overflow_calls) == 1 else gradient

        hook = next(trainer.unwrap_model(real_model).encoder.parameters()).register_hook(overflow_once)
        optimizer = torch.optim.SGD(real_model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cpu", init_scale=65536)
        _, _, attempts = trainer.dry_run_step(real_model, x, valid, dt, y, criterion, optimizer,
                                              scaler, device, False, torch.float16)
        hook.remove()
        assert attempts == 2 and scaler.get_scale() == 32768
    finally:
        dist.destroy_process_group()


class DistributedTemporalMOSTests(unittest.TestCase):
    def test_two_cpu_workers(self):
        with tempfile.TemporaryDirectory(prefix="mos-ddp-test-") as directory:
            mp.spawn(distributed_worker, args=(directory,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
