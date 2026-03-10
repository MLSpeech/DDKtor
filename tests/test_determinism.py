"""
Tests that DDKtor model predictions are deterministic regardless of file
processing order.

The bug: torch.manual_seed is set once globally in predict.py, but
PredictDataset.__init__ only resets np.random.seed and random.seed —
not torch.manual_seed. Since CnnLstmRawSim.init_hidden() calls
nn.init.uniform_() (which draws from torch's RNG), the LSTM hidden
state depends on cumulative RNG consumption from all previously
processed files. Different file orderings → different predictions.
"""

import os
import sys
import numpy as np
import random
from unittest.mock import patch
import torch
import torch.nn.functional as F

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import dataset
from model import load_model

SEED = 1245
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model_cnn_lstm",
    "data_KASIA_ntype_lstm_sim_lr_0.0001_input_size_256_num_layers_2"
    "_hidden_size_256_channels_256_normalize_True_norm_type_z_biLSTM_True"
    "_measure_rval_dropout_0.3_class_num_3_sigmoid_False_chain_bandreject,"
    "noise_lamda_1.0_59021734.pth",
)
DEVICE = "cpu"

# Use two different wav files from sarah-data
SMR_FILE = os.path.join(PROJECT_ROOT, "sarah-data", "4016_02_ddk_alternating_SEPT.wav")
AMR_FILE = os.path.join(PROJECT_ROOT, "sarah-data", "4016_02_ddk_repetition_SEPT.wav")


def _run_prediction(model, wav_path, seed, normalize, norm_type):
    """Run model on a single wav file and return raw logits (pre-softmax)."""
    pred_dataset = dataset.PredictDataset(
        wav_path, seed, slices_size=5000, overlap=0,
        normalize=normalize, norm_type=norm_type,
    )
    pred_loader = torch.utils.data.DataLoader(
        pred_dataset, batch_size=100, shuffle=False,
        num_workers=0, pin_memory=False,
        collate_fn=dataset.PadCollatePred(dim=0),
    )

    all_logits = []
    for raw, lens_list in pred_loader:
        raw = raw.to(DEVICE)
        hidden = model.init_hidden(raw.size(0), DEVICE)
        all_outputs = model(raw, hidden, lens_list)
        output = all_outputs[0]
        for idx in range(output.size(0)):
            cur_len = lens_list[idx]
            all_logits.append(output[idx, :cur_len].detach().cpu())

    return torch.cat(all_logits, dim=0)


def test_order_dependent_predictions_before_fix():
    """Proves the bug: processing a different file first changes logits for
    the target file, because torch RNG state is not reset per-file.

    We patch torch.manual_seed to a no-op inside PredictDataset.__init__
    to simulate the old code that was missing that call.
    """

    model, normalize, sigmoid, norm_type = load_model(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    # Patch torch.manual_seed so dataset.__init__ can't reset it,
    # simulating the pre-fix behavior.
    with patch("dataset.torch.manual_seed"):
        with torch.no_grad():
            # Run A: target file only
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            random.seed(SEED)
            logits_a = _run_prediction(
                model, SMR_FILE, SEED, normalize, norm_type,
            )

            # Run B: dummy file first, then target file
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            random.seed(SEED)
            _run_prediction(
                model, AMR_FILE, SEED, normalize, norm_type,
            )
            logits_b = _run_prediction(
                model, SMR_FILE, SEED, normalize, norm_type,
            )

    # Logits should differ — proving the bug
    assert not torch.allclose(logits_a, logits_b, atol=1e-7), (
        "Expected logits to DIFFER when torch RNG is not reset per-file, "
        "but they were identical. Bug may already be fixed or test setup is wrong."
    )


def test_order_independent_predictions_after_fix():
    """Proves the fix: the real dataset.py now resets torch.manual_seed
    per-file, so logits are identical regardless of processing order."""

    model, normalize, sigmoid, norm_type = load_model(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        # Run A: target file only
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        logits_a = _run_prediction(
            model, SMR_FILE, SEED, normalize, norm_type,
        )

        # Run B: dummy file first, then target
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        _run_prediction(
            model, AMR_FILE, SEED, normalize, norm_type,
        )
        logits_b = _run_prediction(
            model, SMR_FILE, SEED, normalize, norm_type,
        )

    # Logits should be identical — proving the fix works
    assert torch.allclose(logits_a, logits_b, atol=1e-7), (
        "Expected logits to be IDENTICAL when torch RNG is reset per-file, "
        f"but max diff was {(logits_a - logits_b).abs().max().item():.2e}."
    )
