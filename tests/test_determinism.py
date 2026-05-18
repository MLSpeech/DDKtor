"""
Tests that DDKtor model predictions are deterministic regardless of file
processing order.

The bug: torch.manual_seed is set once globally in predict.py, but
PredictDataset.__init__ only resets np.random.seed and random.seed —
not torch.manual_seed. Since CnnLstmRawSim.init_hidden() calls
nn.init.uniform_() (which draws from torch's RNG), the LSTM hidden
state depends on cumulative RNG consumption from all previously
processed files. Different file orderings -> different predictions.
"""

import os
import sys
import tempfile
import numpy as np
import random
from unittest.mock import patch
import torch
import soundfile

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
SR = 16000


def _make_wav(path, duration_sec, rng_seed):
    """Generate a synthetic wav file at 16kHz."""
    rng = np.random.RandomState(rng_seed)
    samples = rng.randn(int(SR * duration_sec)).astype(np.float32) * 0.1
    soundfile.write(path, samples, SR)


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

    with tempfile.TemporaryDirectory() as tmpdir:
        file_a = os.path.join(tmpdir, "a.wav")
        file_b = os.path.join(tmpdir, "b.wav")
        _make_wav(file_a, duration_sec=1.0, rng_seed=42)
        _make_wav(file_b, duration_sec=0.5, rng_seed=99)

        with patch("dataset.torch.manual_seed"):
            with torch.no_grad():
                # Run A: target file only
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                random.seed(SEED)
                logits_a = _run_prediction(
                    model, file_a, SEED, normalize, norm_type,
                )

                # Run B: dummy file first, then target file
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                random.seed(SEED)
                _run_prediction(
                    model, file_b, SEED, normalize, norm_type,
                )
                logits_b = _run_prediction(
                    model, file_a, SEED, normalize, norm_type,
                )

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

    with tempfile.TemporaryDirectory() as tmpdir:
        file_a = os.path.join(tmpdir, "a.wav")
        file_b = os.path.join(tmpdir, "b.wav")
        _make_wav(file_a, duration_sec=1.0, rng_seed=42)
        _make_wav(file_b, duration_sec=0.5, rng_seed=99)

        with torch.no_grad():
            # Run A: target file only
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            random.seed(SEED)
            logits_a = _run_prediction(
                model, file_a, SEED, normalize, norm_type,
            )

            # Run B: dummy file first, then target
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            random.seed(SEED)
            _run_prediction(
                model, file_b, SEED, normalize, norm_type,
            )
            logits_b = _run_prediction(
                model, file_a, SEED, normalize, norm_type,
            )

        assert torch.allclose(logits_a, logits_b, atol=1e-7), (
            "Expected logits to be IDENTICAL when torch RNG is reset per-file, "
            f"but max diff was {(logits_a - logits_b).abs().max().item():.2e}."
        )
