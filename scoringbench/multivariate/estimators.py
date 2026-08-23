"""Sample-space Monte-Carlo estimators shared by the multivariate scoring rules.

Everything here operates directly on ensembles of draws — there is no grid, PMF,
CDF, or density anywhere (that is the whole point of the multivariate package).
The building blocks are:

* :func:`force_precision` — decorator upcasting float tensors to a working
  dtype (copied locally so the multivariate package does not import from
  ``scoringbench.univariate``).
* :func:`pairwise_norm_expectation` — the "term 2" of the energy score,
  ``E‖Y − Y'‖^β`` under the *fair* (unbiased) estimator
  ``1 / (m(m−1)) Σ_{i≠j} ‖yᵢ − yⱼ‖^β``.
* :func:`cross_norm_expectation` — the "term 1" of the energy score,
  ``E‖Y − y‖^β`` = ``1/m Σᵢ ‖yᵢ − y‖^β``.
* :func:`pairwise_abs_pow_expectation` — per-coordinate
  ``E|Yₐ − Y'ₐ|^p`` under the same fair estimator, used by the variogram score.

The pairwise estimators are chunked over the *draw* axis so an ``m × m`` distance
matrix is never fully materialised for large ensembles.
"""

from __future__ import annotations

import functools

import torch

# Draws processed per chunk when accumulating an m×m pairwise term.  Keeps the
# peak intermediate at ~ (chunk × m × d) instead of (m × m × d).
_PAIRWISE_CHUNK = 256


# ---------------------------------------------------------------------------
# Numerical precision (local copy — no univariate import)
# ---------------------------------------------------------------------------

def force_precision(dtype: torch.dtype = torch.float64):
    """Decorator: upcast every floating-point tensor argument to ``dtype``.

    The Monte-Carlo scoring rules form differences of large, nearly-equal
    expectations (energy ``term1 − term2``; variogram ``(observed − expected)²``;
    Dawid–Sebastiani quadratic form + log-determinant).  Evaluated in float32
    these suffer catastrophic cancellation and can violate guarantees such as
    "energy score ≥ 0".  Computing in float64 restores them.  Integer/index
    tensors and non-tensor arguments pass through unchanged.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def cast(x):
                if isinstance(x, torch.Tensor) and x.is_floating_point():
                    return x.to(dtype)
                return x

            new_args = tuple(cast(a) for a in args)
            new_kwargs = {k: cast(v) for k, v in kwargs.items()}
            return func(*new_args, **new_kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Energy-score building blocks (Euclidean norm in R^d)
# ---------------------------------------------------------------------------

@force_precision(torch.float64)
def cross_norm_expectation(samples: torch.Tensor, y: torch.Tensor, beta: float) -> torch.Tensor:
    """``E‖Y − y‖^β`` per test instance (energy-score term 1).

    Parameters
    ----------
    samples : (n_test, m, d) tensor
        Predictive draws.
    y : (n_test, d) tensor
        Observed target vectors.
    beta : float
        Energy-score exponent (``0 < beta < 2``; ``beta = 1`` is the classic
        energy score).

    Returns
    -------
    (n_test,) tensor
        ``1/m Σᵢ ‖yᵢ − y‖^β``.
    """
    diff = samples - y[:, None, :]               # (n_test, m, d)
    dist = torch.linalg.vector_norm(diff, dim=-1)  # (n_test, m)
    return (dist ** beta).mean(dim=1)


@force_precision(torch.float64)
def pairwise_norm_expectation(samples: torch.Tensor, beta: float) -> torch.Tensor:
    """``E‖Y − Y'‖^β`` per test instance under the FAIR estimator (energy term 2).

    The unbiased ("fair") estimator excludes the diagonal ``i = j``:

        1 / (m (m − 1))  Σ_{i ≠ j}  ‖yᵢ − yⱼ‖^β

    (Gneiting & Raftery 2007; the biased ``1/m²`` variant makes the estimated
    energy score not strictly proper for finite ``m``.)

    Parameters
    ----------
    samples : (n_test, m, d) tensor
    beta : float

    Returns
    -------
    (n_test,) tensor
    """
    n_test, m, _ = samples.shape
    if m < 2:
        # No off-diagonal pair exists; term 2 is undefined -> 0 by convention.
        return samples.new_zeros(n_test)

    total = samples.new_zeros(n_test)
    for start in range(0, m, _PAIRWISE_CHUNK):
        stop = min(start + _PAIRWISE_CHUNK, m)
        block = samples[:, start:stop, :]                       # (n_test, c, d)
        diff = block[:, :, None, :] - samples[:, None, :, :]    # (n_test, c, m, d)
        dist = torch.linalg.vector_norm(diff, dim=-1)           # (n_test, c, m)
        total = total + (dist ** beta).sum(dim=(1, 2))
    # Subtract the m self-pairs (distance 0 -> 0 contribution, but explicit for
    # clarity) and normalise by the number of ordered off-diagonal pairs.
    return total / (m * (m - 1))


# ---------------------------------------------------------------------------
# Variogram-score building block (per-coordinate absolute differences)
# ---------------------------------------------------------------------------

@force_precision(torch.float64)
def pairwise_abs_pow_expectation(samples: torch.Tensor, p: float) -> torch.Tensor:
    """``E|Yₐ − Y'ₐ|^p`` per test instance and coordinate pair (a, b).

    Uses the same fair ``1/(m(m−1))`` estimator as
    :func:`pairwise_norm_expectation`, but coordinate-wise: for every ordered
    pair of dimensions ``(a, b)`` it estimates ``E|Y_a − Y'_b|^p`` … no — the
    variogram compares the *same* draw's coordinates, so this returns the matrix

        M[a, b] = E |Y_a − Y_b|^p           (single draw, two coordinates)

    which is the expectation over draws of ``|y_a − y_b|^p``.

    Parameters
    ----------
    samples : (n_test, m, d) tensor
    p : float
        Variogram order (``p = 0.5`` in the benchmark).

    Returns
    -------
    (n_test, d, d) tensor
        ``M[i, a, b] = 1/m Σ_k |samples[i, k, a] − samples[i, k, b]|^p``.
    """
    # For a fixed draw, form |Y_a - Y_b| across coordinate pairs, then average
    # over draws.  This is an expectation over the m draws (not a pairwise-draw
    # estimator), matching the variogram score definition E|Y_a - Y_b|^p.
    a = samples[:, :, :, None]        # (n_test, m, d, 1)
    b = samples[:, :, None, :]        # (n_test, m, 1, d)
    absdiff = (a - b).abs() ** p      # (n_test, m, d, d)
    return absdiff.mean(dim=1)        # (n_test, d, d)
