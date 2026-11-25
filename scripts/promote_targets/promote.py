"""Promote the univariate ScoringBench suite into a *multivariate* one.

Each univariate dataset is a single-target regression table ``(X, y)``. This
protocol turns it into a multi-target problem by **promoting features into
targets**: the features most *conditionally dependent* with the anchor target,
given the remaining features, move into the target block and the rest stay as
inputs. The aim is a suite whose targets are at least as strongly conditionally
dependent as the KCI-screened ``multitarget_openml`` suite, i.e. datasets where
a joint model can beat a product of marginals.

Why conditional (not marginal) dependence
------------------------------------------
A joint model only helps when ``Q(Y_1..Y_d | X) != prod_j Q(Y_j | X)``. A feature
that correlates with ``y`` marginally can be conditionally independent of it once
the other features are known. Candidates are therefore ranked by **partial
distance correlation**: residualise the anchor and the candidate on the
conditioning features (out-of-bag random forest, ridge for wide X) and take the
distance correlation of the two residuals.

Protocol (per dataset)
----------------------
1. Load ``(X, y)`` with the univariate loader (openml / pmlb / keel).
2. Only **continuous** numeric columns can become targets (discrete/lattice
   columns fail the multivariate continuous-target gate). The anchor
   (``target_0``) is ``y`` when it is continuous, otherwise the continuous feature
   with the largest marginal distance correlation to ``y`` (``y`` is then dropped).
3. On the fixed curation split (``ci_screen.curation_split``; the benchmark never
   sees these rows), rank the candidates by partial distance correlation with the
   anchor given every other feature, and take the top ``budget`` (at most
   ``MAX_PROMOTE = 9``, always leaving ``MIN_CONDITIONERS`` features as inputs).
4. **Refine** (``REFINE`` passes): re-rank every candidate against the
   conditioning set the screen will actually use -- all numeric features minus the
   current promoted block -- and stop early when the block no longer changes.
5. Screen the block at ``d = 2, 3, 5, 10`` with the multivariate screen
   (``multivariate_screening.screening.screen_dataset``: quality gates + KCI +
   residual distance-correlation effect size, ``PROMOTE_EFFECT_MIN = 0.15``). The
   whole promoted block is removed from the inputs at every ``d``, the same
   contract as ``native_scoringbench``, so a ``d=2`` run never sees a ``d=3``
   target as a feature.

Outputs (``results/``): ``promoted_scoringbench_datasets.json`` (the suite the
``promoted_scoringbench`` source reads), ``promotion_log.json`` (every decision)
and ``report.md``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Cap BLAS/OpenMP threads before numpy loads. KCI works on <=1000x1000 matrices;
# with one thread per core (the default) it runs ~6x slower from thread contention.
_THREADS = os.environ.get("SLURM_CPUS_PER_TASK", "4")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
sys.path[:0] = [str(_HERE.parents[2]), str(_HERE.parents[1])]  # repo root, scripts/

from multivariate_screening.ci_screen import ALPHA, curation_split, distance_correlation  # noqa: E402
from multivariate_screening.screening import screen_dataset  # noqa: E402
from scoringbench.univariate.datasets import load_dataset  # noqa: E402
from scoringbench.univariate.discreteness import target_discreteness  # noqa: E402

TARGET_DIMS = (2, 3, 5, 10)
MAX_PROMOTE = max(TARGET_DIMS) - 1   # 9 promoted features + anchor = d=10
MIN_CONDITIONERS = 2                 # always leave at least this many features as inputs
PROMOTE_EFFECT_MIN = 0.15            # residual distance-correlation gate
REFINE = 2                           # re-ranking passes against the screen's conditioning set
RANK_CANDIDATES = 20                 # candidates scored by partial correlation (top marginal dcor)
SCREEN_ROW_CAP = 50_000              # subsample huge tables before screening (bounds runtime)
TARGET_LABEL = "__y__"               # column name reserved for the original target
# Use the CPUs SLURM allocated, not every core on the node (avoids oversubscription).
N_JOBS = int(_THREADS)


def _numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Numeric-coercible feature columns, as floats; all-NaN columns dropped."""
    num = X.apply(lambda c: c if pd.api.types.is_numeric_dtype(c)
                  else pd.to_numeric(c, errors="coerce"))
    num = num.loc[:, num.notna().any()]
    return num.astype(float)


def _is_continuous(values: np.ndarray, name: str) -> bool:
    """A column is a usable target iff it is not discrete/lattice (atom-mass ok)."""
    diag = target_discreteness(values, name=name)
    return not (diag.get("is_discrete") and diag.get("rule") != "atom_mass")


def _residual(target: np.ndarray, X: np.ndarray, seed: int) -> np.ndarray:
    """Fast residuals for ranking: out-of-bag random forest, ridge for wide X.

    Only used to rank candidates. The effect size that decides selection is the
    cross-fitted ``residual_distance_correlation`` inside ``screen_dataset``.
    """
    target = np.asarray(target, float).ravel()
    X = np.asarray(X, float)
    if X.shape[1] == 0:
        return target - target.mean()
    if X.shape[1] <= 50:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, bootstrap=True, oob_score=True,
                                   n_jobs=N_JOBS, random_state=seed).fit(X, target)
        resid = target - rf.oob_prediction_
        bad = ~np.isfinite(resid)                 # rows never out-of-bag
        resid[bad] = target[bad] - target.mean()
        return resid
    from sklearn.linear_model import RidgeCV
    m = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(X, target)
    return target - m.predict(X)


def rank_block(cont: list[str], Fc: pd.DataFrame, ycur: np.ndarray, y_cont: bool,
               budget: int, seed: int, refine: int = REFINE) -> list[str]:
    """Anchor plus the top ``budget`` candidates by partial distance correlation.

    Pass 0 conditions each candidate on every other feature. Each refinement pass
    re-scores the candidates conditioning on numeric features minus the current
    promoted block (what the screen uses), until the block is stable.
    """
    if y_cont:
        anchor_key, anchor_name, anchor_vals, cand = TARGET_LABEL, None, ycur, list(cont)
    else:
        anchor_name = max(cont, key=lambda c: distance_correlation(ycur, Fc[c].to_numpy(), seed=seed))
        anchor_key, anchor_vals = anchor_name, Fc[anchor_name].to_numpy()
        cand = [c for c in cont if c != anchor_name]
    if not cand:
        return []
    marginal = {c: distance_correlation(anchor_vals, Fc[c].to_numpy(), seed=seed) for c in cand}
    cand = sorted(cand, key=marginal.get, reverse=True)[:RANK_CANDIDATES]

    def score(exclude: set[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in cand:
            drop = exclude | {c} | ({anchor_name} if anchor_name else set())
            others = Fc[[x for x in Fc.columns if x not in drop]].to_numpy()
            r_a = _residual(anchor_vals, others, seed)
            r_f = _residual(Fc[c].to_numpy(), others, seed + 1)
            out[c] = distance_correlation(r_a, r_f, seed=seed)
        return out

    def top(scores: dict[str, float]) -> list[str]:
        return [anchor_key] + sorted(scores, key=scores.get, reverse=True)[:budget]

    block = top(score(set()))
    for _ in range(refine):
        new = top(score({b for b in block if b != TARGET_LABEL}))
        if new == block:
            break
        block = new
    return block


def screen_block(block: list[str], F: pd.DataFrame, y: pd.Series, name: str,
                 seed: int) -> tuple[dict, list[int]]:
    """Screen the block at every d, with the whole promoted block removed from X."""
    block_feats = [c for c in block if c != TARGET_LABEL]
    X_cond = F[[c for c in F.columns if c not in block_feats]]
    screens: dict[str, dict] = {}
    selected: list[int] = []
    for d in TARGET_DIMS:
        if len(block) < d:
            continue
        Y = pd.concat([(y.rename(TARGET_LABEL) if c == TARGET_LABEL else F[c].rename(c))
                       for c in block[:d]], axis=1)
        try:
            decision, _ = screen_dataset(X_cond, Y, seed=seed, name=f"{name}_d{d}",
                                         effect_min=PROMOTE_EFFECT_MIN)
        except Exception as exc:
            screens[str(d)] = dict(selected=False, reason=f"error:{type(exc).__name__}")
            continue
        screens[str(d)] = dict(selected=bool(decision.get("selected")),
                               reason=decision.get("reason"),
                               kci_p=decision.get("kci_p"),
                               residual_dcor=decision.get("residual_dcor"),
                               n_features=decision.get("n_features"))
        if decision.get("selected"):
            selected.append(d)
    return screens, selected


def promote_dataset(cfg: dict, seed: int, refine: int = REFINE) -> dict:
    """Promote one univariate dataset and screen the block at each d."""
    X, y = load_dataset(cfg)
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    keep = y.notna().to_numpy()
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)
    if len(y) > SCREEN_ROW_CAP:
        idx = np.random.default_rng(seed).choice(len(y), SCREEN_ROW_CAP, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)

    F = _numeric_frame(X)
    yv = y.to_numpy()
    # Continuity is judged on the full column (matching the screen's audit).
    y_cont = _is_continuous(yv, cfg["name"])
    cont = [c for c in F.columns if _is_continuous(F[c].to_numpy(), f"{cfg['name']}:{c}")]
    record = dict(name=cfg["name"], univariate=cfg, n_rows=int(len(y)),
                  n_numeric_features=int(F.shape[1]), n_continuous_candidates=len(cont),
                  original_target_continuous=bool(y_cont))
    if not y_cont and not cont:
        record["reason"] = "discrete_target_and_no_continuous_feature"
        return record

    # Ranking uses the curation split only (disjoint from the benchmark rows).
    cur = curation_split(len(y), seed=seed)
    Fc = F.loc[cur].fillna(F.loc[cur].median(numeric_only=True)).fillna(0.0)
    ycur = yv[cur]

    # Cap the promoted count so MIN_CONDITIONERS features always remain as inputs.
    # A discrete y is dropped and the anchor is itself a feature, using one slot.
    anchor_is_feature = not y_cont
    budget = min(MAX_PROMOTE, F.shape[1] - MIN_CONDITIONERS - (1 if anchor_is_feature else 0))
    min_feats = 2 if anchor_is_feature else 1
    if budget < min_feats:
        record["reason"] = "insufficient_features_for_conditioning"
        return record

    block = rank_block(cont, Fc, ycur, y_cont, budget, seed, refine=refine)
    if len([c for c in block if c != TARGET_LABEL]) < min_feats:
        record["reason"] = "no_promotable_continuous_feature"
        return record

    screens, selected_dims = screen_block(block, F, y, cfg["name"], seed)
    record.update(anchor=block[0], promoted_cols=[c for c in block if c != TARGET_LABEL],
                  all_target_cols=block, screen=screens, selected_dims=selected_dims)
    return record


def _reference_effect_sizes() -> list[float]:
    """residual_dcor of the KCI-selected multitarget_openml suite at d=2, if available."""
    draft = (_HERE.parents[1] / "multivariate_screening" / "results"
             / "kci_continuous_audited_v3" / "pool" / "suite_draft.json")
    if not draft.exists():
        return []
    out = []
    for rec in json.loads(draft.read_text()).get("datasets", []):
        s = rec.get("screen", {})
        if s.get("selected") and s.get("residual_dcor") is not None:
            out.append(float(s["residual_dcor"]))
    return out


def _write_outputs(records: list[dict], output_dir: Path, refine: int) -> int:
    """Write promoted_scoringbench_datasets.json, promotion_log.json and report.md."""
    # The suite is every dataset selected at d=2 (the entry gate), ordered by name.
    suite = sorted((r for r in records if 2 in r.get("selected_dims", [])),
                   key=lambda r: str(r.get("name")))
    manifest = dict(schema=1, n_datasets=len(suite),
                    how_built="Univariate ScoringBench datasets promoted to multivariate by "
                              "moving the most conditionally dependent features into the target "
                              "block (scripts/promote_targets/promote.py), KCI-screened at d=2.",
                    target_dims=list(TARGET_DIMS), alpha=ALPHA, effect_min=PROMOTE_EFFECT_MIN,
                    selection="partial_distance_correlation", refine=refine, datasets=suite)
    (output_dir / "promoted_scoringbench_datasets.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (output_dir / "promotion_log.json").write_text(json.dumps(records, indent=1) + "\n")

    ref = _reference_effect_sizes()
    eff = [r["screen"]["2"]["residual_dcor"] for r in suite
           if r.get("screen", {}).get("2", {}).get("residual_dcor") is not None]
    lines = ["# Promoted multivariate suite", "",
             f"Promoted {len(suite)} datasets (selected at d=2) out of {len(records)} univariate.",
             "Per-d selected counts: " +
             ", ".join(f"d={d}: {sum(d in r.get('selected_dims', []) for r in records)}"
                       for d in TARGET_DIMS), ""]
    if eff:
        lines.append(f"Promoted residual_dcor @d2: median={np.median(eff):.3f} "
                     f"mean={np.mean(eff):.3f} min={min(eff):.3f}")
    if ref:
        lines.append(f"multitarget_openml residual_dcor @d2: median={np.median(ref):.3f} "
                     f"mean={np.mean(ref):.3f} (n={len(ref)})")
    lines += ["", "| dataset | promoted | selected d | kci_p@d2 | residual_dcor@d2 |",
              "| --- | --- | --- | --- | --- |"]
    for r in records:
        s2 = r.get("screen", {}).get("2", {})
        lines.append(f"| {r.get('name')} | {len(r.get('promoted_cols', []))} | "
                     f"{r.get('selected_dims', [])} | {s2.get('kci_p')} | {s2.get('residual_dcor')} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\npromoted suite: {len(suite)} datasets -> {output_dir / 'promoted_scoringbench_datasets.json'}",
          flush=True)
    return len(suite)


def _promote_safe(cfg: dict, seed: int, refine: int) -> dict:
    try:
        return promote_dataset(cfg, seed=seed, refine=refine)
    except Exception as exc:
        return dict(name=cfg.get("name"), univariate=cfg,
                    reason=f"error:{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", type=Path, default=_HERE.parents[2] / "datasets.json",
                        help="Univariate ScoringBench manifest (list of dataset configs).")
    parser.add_argument("--output_dir", type=Path, default=_HERE.parent / "results")
    parser.add_argument("--max_datasets", type=int, default=None,
                        help="Promote only the first N datasets (quick smoke test).")
    parser.add_argument("--dataset_index", type=int, default=None,
                        help="Promote only the dataset at this index; writes one part file "
                             "under <output_dir>/parts/ (for SLURM array jobs).")
    parser.add_argument("--merge", action="store_true",
                        help="Merge every <output_dir>/parts/*.json into the suite manifest + report.")
    parser.add_argument("--refine", type=int, default=REFINE,
                        help=f"Re-ranking passes against the screen's conditioning set "
                             f"(default {REFINE}).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        parts = sorted((args.output_dir / "parts").glob("promote_*.json"))
        records = [json.loads(p.read_text()) for p in parts]
        n = _write_outputs(records, args.output_dir, args.refine)
        print(f"merged {len(records)} parts -> {n} selected datasets", flush=True)
        return 0

    configs = json.loads(args.datasets.read_text())

    if args.dataset_index is not None:        # SLURM array mode: one part file per task
        cfg = configs[args.dataset_index]
        rec = _promote_safe(cfg, args.seed, args.refine)
        parts_dir = args.output_dir / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        (parts_dir / f"promote_{args.dataset_index:04d}.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(f"[{args.dataset_index}] {cfg.get('name'):35s} "
              f"promoted={len(rec.get('promoted_cols', []))} selected_d={rec.get('selected_dims', [])} "
              f"{rec.get('reason', '')}", flush=True)
        return 0

    if args.max_datasets is not None:
        configs = configs[: args.max_datasets]
    records = []
    for i, cfg in enumerate(configs):
        rec = _promote_safe(cfg, args.seed, args.refine)
        records.append(rec)
        print(f"[{i + 1}/{len(configs)}] {cfg.get('name'):35s} "
              f"promoted={len(rec.get('promoted_cols', []))} selected_d={rec.get('selected_dims', [])} "
              f"{rec.get('reason', '')}", flush=True)
    _write_outputs(records, args.output_dir, args.refine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
