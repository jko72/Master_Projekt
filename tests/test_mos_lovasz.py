"""Loss edge cases, upstream equivalence, integration and accumulation."""

import copy
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "tests")]
from helper.mos_lovasz_loss import MOSLovaszSoftmax, MOSCrossEntropyLovasz, loss_settings
from reference.lmnet_lovasz import lovasz_softmax as reference_lovasz
import train_temporal_mos_e2e as trainer
from test_temporal_mos_distributed import TinyDataset, TinyModel, batch


class LovaszTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_labels_ignore_extremes_and_backward(self):
        for labels in ([[0, 0, 0, 0]], [[1, 1, 1, 1]], [[0, 1, 0, 1]],
                       [[-1, 0, 1, -1]], [[-1, 1, -1, -1]], [[-1, -1, -1, -1]]):
            for classes in ("present", "all"):
                for extreme in (False, True):
                    with self.subTest(labels=labels, classes=classes, extreme=extreme):
                        logits = (torch.randn(1, 2, 1, 4) * (10000 if extreme else 1)).requires_grad_()
                        target = torch.tensor(labels).reshape(1, 1, 4)
                        loss = MOSLovaszSoftmax(classes)(logits, target)
                        self.assertEqual(loss.shape, ())
                        self.assertEqual(loss.dtype, torch.float32)
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        self.assertTrue(torch.isfinite(logits.grad).all())
                        self.assertTrue((logits.grad.movedim(1, -1)[target == -1] == 0).all())
                        if (target == -1).all():
                            self.assertEqual(loss.item(), 0)

    def test_single_pixel_exact(self):
        logits = torch.tensor([[[[0.0]], [[0.0]]]], requires_grad=True)
        loss = MOSLovaszSoftmax()(logits, torch.ones(1, 1, 1, dtype=torch.long))
        torch.testing.assert_close(loss, torch.tensor(0.5))
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[[[0.25]], [[-0.25]]]]))

    def test_original_lmnet_reference_value_and_gradients(self):
        for seed in range(4):
            for classes in ("present", "all"):
                for per_image in (False, True):
                    torch.manual_seed(seed)
                    logits = torch.randn(2, 2, 3, 4, requires_grad=True)
                    reference_logits = logits.detach().clone().requires_grad_()
                    target = torch.randint(-1, 2, (2, 3, 4))
                    actual = MOSLovaszSoftmax(classes, per_image)(logits, target)
                    expected = reference_lovasz(reference_logits.softmax(1), target,
                                                classes=classes, per_image=per_image, ignore=-1)
                    torch.testing.assert_close(actual, expected)
                    actual.backward()
                    expected.backward()
                    torch.testing.assert_close(logits.grad, reference_logits.grad)

    def test_per_image_with_ignored_image(self):
        logits = torch.randn(2, 2, 2, 3, requires_grad=True)
        target = torch.zeros(2, 2, 3, dtype=torch.long)
        target[0] = -1
        expected = MOSLovaszSoftmax()(logits[1:], target[1:]) / 2
        torch.testing.assert_close(MOSLovaszSoftmax(per_image=True)(logits, target), expected)

    def test_ce_regression_and_combination(self):
        weights = torch.tensor([1.0, 10.0])
        cfg = {"mos_train_params": dict(static_class_weight=1, moving_class_weight=10)}
        criterion = trainer.make_criterion(cfg, torch.device("cpu"))
        self.assertIs(type(criterion), torch.nn.CrossEntropyLoss)
        logits = torch.randn(2, 2, 3, 4, requires_grad=True)
        target = torch.randint(-1, 2, (2, 3, 4))
        expected_ce = torch.nn.functional.cross_entropy(logits, target, weight=weights, ignore_index=-1)
        torch.testing.assert_close(criterion(logits, target), expected_ce, rtol=0, atol=0)
        for weight in (0.0, 1.0, 0.3):
            combined = MOSCrossEntropyLovasz(weights, weight)
            parts = combined.components(logits, target)
            torch.testing.assert_close(parts["loss_ce"], expected_ce, rtol=0, atol=0)
            torch.testing.assert_close(parts["loss_total"], expected_ce + weight * parts["loss_lovasz"])
            if weight == 0:
                torch.testing.assert_close(torch.autograd.grad(parts["loss_total"], logits, retain_graph=True)[0],
                                           torch.autograd.grad(expected_ce, logits, retain_graph=True)[0], rtol=0, atol=0)
        parts = combined.components(logits, torch.full_like(target, -1))
        self.assertEqual(parts["loss_total"].item(), 0)
        parts["loss_total"].backward()
        self.assertTrue((logits.grad == 0).all())

    def test_configuration_and_ablation_isolation(self):
        self.assertEqual(loss_settings()["type"], "ce")
        invalid = [[], {"type": "other"}, {"lovasz_weight": -1}, {"lovasz_weight": float("nan")},
                   {"lovasz_weight": float("inf")}, {"lovasz_weight": True},
                   {"lovasz_classes": "moving"}, {"lovasz_per_image": "false"}, {"typo": 1}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                loss_settings(value)
        _, baseline = trainer.read_config(ROOT / "src/configs/mos_semanticKitti_temporal_e2e_intensity_off8_p2_fullres_2gpu.yaml")
        _, ablation = trainer.read_config(ROOT / "src/configs/mos_semanticKitti_temporal_e2e_intensity_off8_fullres_lovasz.yaml")
        self.assertEqual(ablation["mos_train_params"].pop("loss")["type"], "ce_lovasz")
        baseline["mos_train_params"].pop("loss")
        ablation["mos_log_params"].pop("run_name")
        baseline["mos_log_params"].pop("run_name")
        self.assertEqual(baseline, ablation)

    def test_invalid_labels_and_shape(self):
        with self.assertRaisesRegex(ValueError, "MOS target"):
            MOSLovaszSoftmax()(torch.randn(1, 2, 1, 1), torch.tensor([[[2]]]))
        with self.assertRaisesRegex(ValueError, "Expected MOS"):
            MOSLovaszSoftmax()(torch.randn(1, 3, 1, 1), torch.zeros(1, 1, 1, dtype=torch.long))

    def test_cpu_autocast_fp32_loss(self):
        logits = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16, requires_grad=True)
        target = torch.randint(0, 2, (1, 3, 4))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = MOSLovaszSoftmax()(logits, target)
        self.assertEqual(actual.dtype, torch.float32)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_fp16_amp(self):
        logits = torch.randn(2, 2, 16, 32, device="cuda", dtype=torch.float16, requires_grad=True)
        target = torch.randint(-1, 2, (2, 16, 32), device="cuda")
        criterion = MOSCrossEntropyLovasz(torch.tensor([1., 10.], device="cuda"))
        with torch.autocast("cuda", dtype=torch.float16):
            parts = criterion.components(logits, target)
        self.assertEqual(parts["loss_lovasz"].dtype, torch.float32)
        parts["loss_total"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_accumulation_matches_microbatch_objective(self):
        # Unequal CE denominators deliberately verify the existing average of
        # microbatch CE means, plus equally averaged local Lovasz, not a new
        # concatenated-batch objective. Final accumulation group has size one.
        torch.manual_seed(42)
        model = TinyModel()
        reference = copy.deepcopy(model)
        dataset = TinyDataset(cfg={"test_samples": 5})
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        criterion = MOSCrossEntropyLovasz(torch.tensor([1., 10.]))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        for indices in (([0, 1], [2, 3]), ([4],)):
            ref_optimizer.zero_grad()
            for microbatch in indices:
                x, valid, dt, y = batch(dataset, microbatch)
                (criterion(reference(x, valid, dt), y) / len(indices)).backward()
            ref_optimizer.step()
        result = trainer.run_phase(model, loader, criterion, torch.device("cpu"), optimizer,
                                   torch.amp.GradScaler("cuda", enabled=False), accumulation=2, clip=0)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)
        self.assertAlmostEqual(result["loss_total"], result["loss_ce"] + result["loss_lovasz"])


if __name__ == "__main__":
    unittest.main()
