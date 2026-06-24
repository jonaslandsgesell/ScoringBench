"""Regression test: CRPS / energy score must be non-negative.

The histogram energy-score estimator (used for CRPS at beta=1.0) computes
``term1 - term2`` — a difference of potentially large, nearly-equal values.
In float32 the catastrophic cancellation drove CRPS as low as -5.8 on sharp,
wide-binned quantile-histogram predictions (e.g. 999-quantile model heads),
which spuriously lowered those models' mean CRPS. The estimator now runs in
float64 and clamps per-sample to >= 0; these tests guard against regressions.
"""
import numpy as np
import torch

from scoringbench.metrics import compute_energy_score_histogram_corrected


def _crps(probas, mids, widths, y, dtype=torch.float32):
    out = compute_energy_score_histogram_corrected(
        torch.as_tensor(probas, dtype=dtype),
        torch.as_tensor(mids, dtype=dtype),
        torch.as_tensor(widths, dtype=dtype),
        torch.as_tensor(y, dtype=dtype),
        betas=[1.0],
    )
    return out["energy_score_beta_1.0"]


def test_crps_nonnegative_large_scale_sharp():
    """Large-scale target + sharp many-bin distribution — the float32 regime
    where term1 - term2 cancellation used to go negative."""
    K = 1001
    mids = np.linspace(1.0e5 - 50.0, 1.0e5 + 50.0, K)
    widths = np.full(K, 100.0 / (K - 1))
    probas = (np.ones(K) / K)[None, :]
    for y in (1.0e5, 1.0e5 + 5.0, 1.0e5 - 30.0):
        assert _crps(probas, mids, widths, [y]) >= 0.0


def test_crps_nonnegative_random_histograms():
    """Random valid histograms over a large dynamic range stay non-negative in
    both float32 and float64."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        edges = np.unique(np.sort(rng.normal(scale=1e4, size=257)))
        if edges.size < 3:
            continue
        mids = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges)
        p = rng.random(mids.size)
        p /= p.sum()
        y = float(rng.normal(scale=1e4))
        for dtype in (torch.float32, torch.float64):
            assert _crps(p[None, :], mids, widths, [y], dtype=dtype) >= 0.0
