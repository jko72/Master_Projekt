#!/usr/bin/env python3
# @brief: Simplified Base Model (no Lightning, no utils dependency)

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np

# --- Optional Fallbacks für fehlende Pakete ---
try:
    import lightning.pytorch as pl
except ImportError:
    # Fallback: einfache Ersatzklasse, falls Lightning nicht installiert ist
    class _DummyModule(nn.Module):
        def save_hyperparameters(self, *args, **kwargs): pass
    pl = type("pl", (), {"LightningModule": _DummyModule})

# Dummy-Klassen für fehlende Utility-Module (werden hier NICHT genutzt)
class DummyProjection:
    def __init__(self, cfg): pass
    def __call__(self, *args, **kwargs): pass

class DummyLogger:
    def __init__(self): pass
    def log_point_clouds(self, *args, **kwargs): pass
    def save_range_and_mask(self, *args, **kwargs): pass
    def save_point_clouds(self, *args, **kwargs): pass


# =============================================================================
# Vereinfachte BasePredictionModel (Lightning-frei)
# =============================================================================
class BasePredictionModel(pl.LightningModule):
    """Base class for ACC models (simplified, Lightning-free)"""

    def __init__(self, cfg):
        super(BasePredictionModel, self).__init__()
        self.cfg = cfg
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters(self.cfg)

        # --- Data config fallback ---
        data_cfg = cfg.get("DATA_CONFIG", {})
        self.height = data_cfg.get("HEIGHT", 64)
        self.width = data_cfg.get("WIDTH", 512)
        self.min_range = data_cfg.get("MIN_RANGE", 0.0)
        self.max_range = data_cfg.get("MAX_RANGE", 80.0)
        mean = data_cfg.get("MEAN", [0.0])
        std = data_cfg.get("STD", [1.0])
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

        # --- Model info ---
        model_cfg = cfg.get("MODEL", {})
        self.n_past_steps = model_cfg.get("N_PAST_STEPS", 5)
        self.n_future_steps = model_cfg.get("N_FUTURE_STEPS", 5)
        use_cfg = model_cfg.get("USE", {"XYZ": False, "INTENSITY": False})
        self.use_xyz = use_cfg.get("XYZ", False)
        self.use_intensity = use_cfg.get("INTENSITY", False)

        # --- I/O Info ---
        self.inputs = [0]
        if self.use_xyz:
            self.inputs.extend([1, 2, 3])
        if self.use_intensity:
            self.inputs.append(4)
        self.n_inputs = len(self.inputs)

        # Placeholder attributes (nicht zwingend benötigt)
        self.projection = DummyProjection(cfg)
        self.logger = DummyLogger()
        self.chamfer_distances_tensor = torch.zeros(self.n_future_steps, 1)

    # -------------------------------------------------------------
    # Diese Methoden werden im aktuellen Projekt NICHT aufgerufen,
    # bleiben aber kompatibel, falls du sie später brauchst.
    # -------------------------------------------------------------
    def forward(self, x):
        raise NotImplementedError("BasePredictionModel.forward() should be implemented in derived classes.")

    def configure_optimizers(self):
        """Standard-Optimizer; Lightning-kompatibel, aber auch ohne nutzbar."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=1.0)
        return [optimizer], [scheduler]
