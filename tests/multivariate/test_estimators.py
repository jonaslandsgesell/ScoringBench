"""Unit tests for the sample-space Monte-Carlo estimators.

These verify the *building blocks* of the multivariate scoring rules directly
(energy term 1 / term 2, and the variogram absolute-difference matrix), against
brute-force reference implementations and known closed forms.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scoringbench.multivariate.estimators import (
    cross_norm_expectation,
    force_precision,
    pairwise_abs_pow_expectation,
    pairwise_norm_expectation,
)


# ---------------------------------------------------------------------------
# Brute-force references
# ---------------------------------------------------------------------------

def _ref_cross(samples, y, beta):
    """Naive E‖Y − y‖^β = 1/m Σ_i ‖y_i − y‖^β."""
    n_test = samples.shape[0]
    out = np.empty(n_test)
    for i in range(n_test):
        d = np.linalg.norm(samples[i] - y[i][None, :], axis=-1)
        out[i] = np.mean(d ** beta)
    return out


def _ref_pairwise(samples, beta):
    """Naive fair estimator 1/(m(m-1)) Σ_{i≠j} ‖y_i − y_j‖^β."""
    n_test, m, _ = samples.shape
    if m < 2:
        return np.zeros(n_test)
    out = np.empty(n_test)
    for t in range(n_test):
        acc = 0.0
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                acc += np.linalg.norm(samples[t, i] - samples[t, j]) ** beta
        out[t] = acc / (m * (m - 1))
    return out


def _ref_abs_pow(samples, p):
    """Naive M[t,a,b] = 1/m Σ_k |y_{k,a} − y_{k,b}|^p."""
    n_test, m, d = samples.shape
    out = np.empty((n_test, d, d))
    for t in range(n_test):
        for a in range(d):
            for b in range(d):
                out[t, a, b] = np.mean(np.abs(samples[t, :, a] - samples[t, :, b]) ** p)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta", [0.5, 1.0, 1.5])
def test_cross_norm_matches_reference(beta):
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(7, 40, 3))
    y = rng.normal(size=(7, 3))
    got = cross_norm_expectation(
        torch.as_tensor(samples), torch.as_tensor(y), beta
    ).numpy()
    assert np.allclose(got, _ref_cross(samples, y, beta), atol=1e-10)


@pytest.mark.parametrize("beta", [0.5, 1.0, 1.5])
def test_pairwise_norm_matches_reference(beta):
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(5, 30, 2))
    got = pairwise_norm_expectation(torch.as_tensor(samples), beta).numpy()
    assert np.allclose(got, _ref_pairwise(samples, beta), atol=1e-10)


def test_pairwise_norm_chunking_is_exact():
    """Chunked accumulation must equal a single-chunk computation."""
    from scoringbench.multivariate import estimators as est

    rng = np.random.default_rng(2)
    samples = torch.as_tensor(rng.normal(size=(3, 600, 4)))  # m > _PAIRWISE_CHUNK
    chunked = pairwise_norm_expectation(samples, 1.0).numpy()
    ref = _ref_pairwise(samples.numpy(), 1.0)
    assert np.allclose(chunked, ref, atol=1e-9)


def test_pairwise_norm_single_draw_is_zero():
    samples = torch.zeros((4, 1, 3))
    out = pairwise_norm_expectation(samples, 1.0)
    assert torch.all(out == 0.0)


def test_pairwise_norm_diagonal_excluded():
    """Two identical draws: only the off-diagonal pair contributes."""
    # draws = [a, b]; distance a-b appears twice (ordered), /(m(m-1))=2 -> |a-b|.
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])  # ‖a-b‖ = 5
    samples = torch.as_tensor(np.stack([a, b])[None, :, :])  # (1, 2, 2)
    out = pairwise_norm_expectation(samples, 1.0)
    assert out.item() == pytest.approx(5.0)


@pytest.mark.parametrize("p", [0.5, 1.0, 2.0])
def test_abs_pow_matches_reference(p):
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(6, 25, 4))
    got = pairwise_abs_pow_expectation(torch.as_tensor(samples), p).numpy()
    assert np.allclose(got, _ref_abs_pow(samples, p), atol=1e-10)


def test_abs_pow_diagonal_is_zero():
    """|Y_a − Y_a|^p = 0, so the diagonal of the matrix must vanish."""
    rng = np.random.default_rng(4)
    samples = torch.as_tensor(rng.normal(size=(3, 10, 5)))
    M = pairwise_abs_pow_expectation(samples, 0.5).numpy()
    assert np.allclose(np.diagonal(M, axis1=1, axis2=2), 0.0, atol=1e-12)


def test_abs_pow_symmetric():
    rng = np.random.default_rng(5)
    samples = torch.as_tensor(rng.normal(size=(3, 10, 4)))
    M = pairwise_abs_pow_expectation(samples, 1.0).numpy()
    assert np.allclose(M, np.transpose(M, (0, 2, 1)), atol=1e-12)


def test_force_precision_upcasts_float32():
    @force_precision(torch.float64)
    def f(x):
        return x

    out = f(torch.ones(3, dtype=torch.float32))
    assert out.dtype == torch.float64


def test_force_precision_passes_through_non_float():
    @force_precision(torch.float64)
    def f(x, k):
        return x, k

    _, k = f(torch.ones(3), 5)
    assert k == 5  # int argument untouched
