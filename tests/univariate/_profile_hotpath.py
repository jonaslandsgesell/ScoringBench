"""Ad-hoc profiler for the univariate scoring hot path.

Run:  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python tests/univariate/_profile_hotpath.py
Not a test (no ``test_`` prefix) -- pytest will not collect it.
"""
import cProfile
import io
import logging
import pstats
import time

import numpy as np

logging.disable(logging.CRITICAL)

from scoringbench.univariate.metrics import compute_scoring_rules  # noqa: E402
from scoringbench.univariate.wrappers.base import DistributionPrediction  # noqa: E402

NB = 8


def spike(centres, eps, n_bins=NB):
    centres = np.asarray(centres, dtype=float)
    offs = np.linspace(-0.5, 0.5, n_bins + 1)[None, :] * eps
    edges = centres[:, None] + offs
    return DistributionPrediction(
        probas=np.full((len(centres), n_bins), 1.0 / n_bins),
        bin_edges=edges,
        bin_midpoints=0.5 * (edges[:, :-1] + edges[:, 1:]),
        mean=centres,
        train_range=(float(edges.min()), float(edges.max())),
    )


def main():
    for n in (2_000, 40_000):
        rng = np.random.default_rng(7)
        ys = np.where(rng.random(n) < 0.4, 0.5, rng.uniform(0.0, 1.0, n))
        centres = np.full(n, 0.5)
        d = spike(centres, 1e-1)
        t = time.perf_counter()
        compute_scoring_rules(d, ys)
        print(f"n={n:>6}  one scoring call: {time.perf_counter() - t:.2f}s")

    n = 40_000
    rng = np.random.default_rng(7)
    ys = np.where(rng.random(n) < 0.4, 0.5, rng.uniform(0.0, 1.0, n))
    centres = np.full(n, 0.5)

    pr = cProfile.Profile()
    pr.enable()
    compute_scoring_rules(spike(centres, 1e-4), ys)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(25)
    print(s.getvalue())


if __name__ == "__main__":
    main()
