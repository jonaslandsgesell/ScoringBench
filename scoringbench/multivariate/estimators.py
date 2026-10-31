"""Sample-space Monte-Carlo estimators shared by the multivariate scoring rules.

Everything here operates directly on ensembles of draws — there is no grid, PMF,
CDF, or density anywhere (that is the whole point of the multivariate package).
The building blocks are:

* :func:`force_precision` — decorator upcasting float tensors to a working
  dtype (copied locally so the multivariate package does not import from
  ``scoringbench.univariate``).
* :func:`pairwise_norm_expectation` — the "term 2" of the energy score,
    ``E‖Y − Y'‖^β`` under the unbiased estimator
  ``1 / (m(m−1)) Σ_{i≠j} ‖yᵢ − yⱼ‖^β``.
* :func:`cross_norm_expectation` — the "term 1" of the energy score,
  ``E‖Y − y‖^β`` = ``1/m Σᵢ ‖yᵢ − y‖^β``.
* :func:`pairwise_abs_pow_expectation` — within-draw coordinate differences
    ``E|Yₐ − Y_b|^p``, used by the variogram score.

The pairwise estimators are chunked over the *draw* axis so an ``m × m`` distance
matrix is never fully materialised for large ensembles.
"""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=1)
def _compute_device() -> torch.device:
    """Device the Monte-Carlo scoring estimators run on, resolved ONCE.

    Mirrors ``scoringbench.univariate.metrics._compute_device``: the heavy
    pairwise energy term is a batched matmul that is 6–16× faster on CUDA (see
    :func:`pairwise_norm_expectation`), so the scoring rules default to the GPU
    when one is present.  ``torch.cuda.is_available()`` queries the CUDA driver
    (not free), and the answer cannot change within a process, so cache it.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Peak-element budget for the largest temporary materialised by the pairwise
# energy term.  The Gram-based estimator (below) forms a ``(rows × block × m)``
# distance tensor and a few temporaries of that shape, so the true peak is a
# small multiple of this.  32M float64 ≈ 256 MB per temporary keeps the peak
# comfortably within an 8 GB GPU while staying large enough to keep the matmul
# efficient; it also bounds CPU memory, where the old (rows × block × m × d)
# intermediate used to OOM for moderate d.
_PAIRWISE_ELEM_BUDGET = 32_000_000


# ---------------------------------------------------------------------------
# Numerical precision (local copy — no univariate import)
# ---------------------------------------------------------------------------

def force_precision(dtype: torch.dtype = torch.float64):
    """Decorator: upcast every floating-point tensor argument to ``dtype``.

    The Monte-Carlo scoring rules form differences of large, nearly-equal
    expectations (energy ``term1 − term2``; variogram ``(observed − expected)²``;
    Dawid–Sebastiani quadratic form + log-determinant).  Evaluated in float32
    these can suffer catastrophic cancellation. Float64 reduces roundoff, but
    a finite-sample energy-score estimate can still legitimately be negative.
    Integer/index tensors and non-tensor arguments pass through unchanged.
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


def _pairwise_chunking(n_test: int, m: int) -> tuple[int, int]:
    """Row- and draw-block sizes that keep each temporary within the budget.

    The pairwise term forms, per row block of ``rb`` instances and draw block of
    ``c`` draws, a ``(rb, c, m)`` distance tensor.  We size ``rb`` and ``c`` so
    ``rb * c * m <= _PAIRWISE_ELEM_BUDGET``: if a single row's full ``m × m``
    matrix fits, we batch whole rows (``c = m``); otherwise we process one row at
    a time and chunk its draws.
    """
    per_row = m * m
    if per_row <= _PAIRWISE_ELEM_BUDGET:
        rows = max(1, min(n_test, _PAIRWISE_ELEM_BUDGET // per_row))
        return rows, m
    return 1, max(1, _PAIRWISE_ELEM_BUDGET // m)


@force_precision(torch.float64)
def pairwise_norm_expectation(samples: torch.Tensor, beta: float) -> torch.Tensor:
    """Unbiased ``E‖Y − Y'‖^β`` per test instance (energy-score term 2).

    The estimator excludes the diagonal ``i = j``:

        1 / (m (m − 1))  Σ_{i ≠ j}  ‖yᵢ − yⱼ‖^β

    (Gneiting & Raftery 2007; the biased ``1/m²`` variant makes the estimated
    energy score not strictly proper for finite ``m``.)

    Implementation
    --------------
    This O(m²) term dominates the multivariate scoring cost, so it is computed
    from the Gram identity ``‖a − b‖² = ‖a‖² + ‖b‖² − 2 a·b``: the ``a·b`` block
    is a single batched matmul (``torch.bmm``), which is far faster on CUDA than
    materialising the ``(rows, block, m, d)`` coordinate-difference tensor the
    naive norm requires — and, crucially, it never forms that ``d``-axis
    intermediate, so cost and peak memory are *independent of* ``d``.  Measured
    on an RTX 2000 Ada (float64): ~6–16× faster than the CPU norm at ``m = 600``
    and, unlike the naive path, it does not OOM at moderate ``d``.

    The result matches the naive estimator up to floating-point associativity
    (≈ 1e-12 relative for random ensembles; bit-exact for integer-valued draws,
    where the dot products are exact in float64).  The squared distance is
    clamped at 0 before the square root to absorb the tiny negative values the
    Gram identity can produce near coincident draws.

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

    sq = (samples * samples).sum(dim=-1)                        # (n_test, m) = ‖yᵢ‖²
    total = samples.new_zeros(n_test)
    row_block, draw_block = _pairwise_chunking(n_test, m)
    for r0 in range(0, n_test, row_block):
        r1 = min(r0 + row_block, n_test)
        rows = samples[r0:r1]                                   # (rb, m, d)
        rows_t = rows.transpose(1, 2)                           # (rb, d, m)
        sq_rows = sq[r0:r1]                                     # (rb, m)
        acc = samples.new_zeros(r1 - r0)
        for c0 in range(0, m, draw_block):
            c1 = min(c0 + draw_block, m)
            block = rows[:, c0:c1, :]                           # (rb, c, d)
            gram = torch.bmm(block, rows_t)                     # (rb, c, m) = ⟨yᵢ, yⱼ⟩
            d2 = sq_rows[:, c0:c1, None] + sq_rows[:, None, :] - 2.0 * gram
            # Self-pairs (i = j) are excluded from the estimator and must be
            # exactly 0: the Gram identity leaves a tiny rounding residual on the
            # diagonal (‖yᵢ‖² computed two ways), which ``dist ** beta`` would
            # amplify for beta < 1.  Pin the diagonal to 0 before the root.
            local = torch.arange(c1 - c0, device=samples.device)
            d2[:, local, c0 + local] = 0.0
            dist = d2.clamp_min(0.0).sqrt()                     # (rb, c, m)
            acc = acc + (dist ** beta).sum(dim=(1, 2))
        total[r0:r1] = acc
    # Self-pairs (i = j) contribute distance 0; normalise by the number of
    # ordered off-diagonal pairs.
    return total / (m * (m - 1))


# ---------------------------------------------------------------------------
# Variogram-score building block (per-coordinate absolute differences)
# ---------------------------------------------------------------------------

@force_precision(torch.float64)
def pairwise_abs_pow_expectation(samples: torch.Tensor, p: float) -> torch.Tensor:
    """``E|Y_a − Y_b|^p`` per test instance and coordinate pair (a, b).

    The variogram compares the same draw's coordinates, so this returns

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
