import pandas as pd
import pytest

from autorank_leaderboard import (
    COVERAGE_LEVELS,
    _with_amce,
    load_metric_long_format,
    load_metric_matrix,
    rank_with_mean_std,
)


@pytest.mark.parametrize("target", [0.2, 0.4, 0.6, 0.8, 0.9, 0.95])
def test_coverage_takes_absolute_error_before_averaging_folds(tmp_path, target):
    metric = f"coverage_{round(target * 100)}"
    for model, coverages in {
        "centered": [target - 0.04, target + 0.04],
        "biased": [target + 0.02, target + 0.02],
    }.items():
        rows = [
            {"dataset": f"d{dataset}", "fold": fold, metric: coverage}
            for dataset in range(5)
            for fold, coverage in enumerate(coverages)
        ]
        pd.DataFrame(rows).to_parquet(tmp_path / f"{model}.parquet", index=False)

    errors, _ = load_metric_matrix(tmp_path, metric, coverage_target=target)
    assert errors["centered"].tolist() == pytest.approx([0.04] * 5)
    assert errors["biased"].tolist() == pytest.approx([0.02] * 5)

    raw, _ = load_metric_long_format(tmp_path, metric)
    original = raw.copy(deep=True)
    actual = rank_with_mean_std(raw, coverage_target=target)
    # The two branches must start their magnitude/rank comparisons from
    # the same dataset-level errors.
    expected = rank_with_mean_std(
        errors.rename_axis(columns="model").stack().rename("score").reset_index(),
        hib=False,
    )
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(raw, original)
    assert actual["model"].tolist() == ["biased", "centered"]

    raw_pivot, _ = load_metric_matrix(tmp_path, metric)
    formula_two_errors = (raw_pivot - target).abs()
    assert formula_two_errors["centered"].tolist() == pytest.approx([0.0] * 5)
    formula_two = rank_with_mean_std(
        formula_two_errors.rename_axis(columns="model").stack().rename("score").reset_index(),
        hib=False,
    )
    assert formula_two["model"].tolist() == ["centered", "biased"]


@pytest.mark.parametrize("hib,first", [(True, "high"), (False, "low")])
def test_noncoverage_ordering_unchanged(hib, first):
    raw = pd.DataFrame([
        {"dataset": dataset, "model": model, "fold": fold, "score": value}
        for dataset in range(5)
        for model, value in [("low", 1.0), ("high", 2.0)]
        for fold in range(2)
    ])
    ranked = rank_with_mean_std(raw, hib=hib)
    assert ranked.iloc[0]["model"] == first
    assert ranked["n_datasets"].tolist() == [5, 5]


def test_amce_is_derived_per_fold_then_averaged_without_rewriting(tmp_path):
    for model, offsets in {"variable": [-0.04, 0.04], "biased": [0.02, 0.02]}.items():
        pd.DataFrame([
            {"dataset": "toy", "fold": fold,
             **{f"coverage_{level}": level / 100 + offset for level in COVERAGE_LEVELS}}
            for fold, offset in enumerate(offsets)
        ]).to_parquet(tmp_path / f"{model}.parquet", index=False)
    before = {path: path.read_bytes() for path in tmp_path.glob("*.parquet")}
    matrix, _ = load_metric_matrix(tmp_path, "amce")
    assert matrix.loc["toy", "variable"] == pytest.approx(0.04)
    assert matrix.loc["toy", "biased"] == pytest.approx(0.02)
    long, _ = load_metric_long_format(tmp_path, "amce")
    expected = long.groupby(["dataset", "model"])["score"].mean().unstack()
    pd.testing.assert_frame_equal(matrix, expected)
    ranking = rank_with_mean_std(long, hib=False)
    assert ranking["model"].tolist() == ["biased", "variable"]
    for path, content in before.items():
        assert path.read_bytes() == content


def test_amce_uses_all_levels_and_does_not_mutate_input():
    empirical = [0.1, 0.5, 0.6, 0.7, 0.95, 1.0]
    raw = pd.DataFrame({f"coverage_{level}": [value]
                        for level, value in zip(COVERAGE_LEVELS, empirical)})
    original = raw.copy(deep=True)
    derived = _with_amce(raw)
    assert derived["amce"].iloc[0] == pytest.approx(0.4 / 6)
    pd.testing.assert_frame_equal(raw, original)
    # Recompute rather than trusting a stale stored score.
    assert _with_amce(raw.assign(amce=99))["amce"].iloc[0] == pytest.approx(0.4 / 6)


def test_amce_missing_levels_are_not_silently_averaged(caplog):
    raw = pd.DataFrame({f"coverage_{level}": [level / 100] for level in COVERAGE_LEVELS})
    raw.loc[0, "coverage_60"] = float("nan")
    assert pd.isna(_with_amce(raw)["amce"].iloc[0])
    assert pd.isna(_with_amce(raw.drop(columns="coverage_60"))["amce"].iloc[0])
    assert "all six coverage levels are required" in caplog.text
    assert "amce" not in _with_amce(pd.DataFrame({"rmse": [1.0]}))


@pytest.mark.parametrize("invalid", [-0.1, 1.1, float("inf")])
def test_amce_rejects_invalid_coverage(invalid):
    raw = pd.DataFrame({f"coverage_{level}": [level / 100] for level in COVERAGE_LEVELS})
    raw.loc[0, "coverage_40"] = invalid
    with pytest.raises(ValueError, match="fractions"):
        _with_amce(raw)


def test_main_renames_coverage_outputs_but_reads_raw_columns(tmp_path, monkeypatch, capsys):
    import json
    import sys
    from types import SimpleNamespace
    import autorank_leaderboard as leaderboard

    for model, offset in [("a", 0.01), ("b", 0.02)]:
        pd.DataFrame([
            {"dataset": f"d{dataset}", "fold": fold, "rmse": 1 + offset,
             **{f"coverage_{level}": level / 100 + offset for level in COVERAGE_LEVELS}}
            for dataset in range(5)
            for fold in range(2)
        ]).to_parquet(tmp_path / f"{model}.parquet", index=False)
    before = {path: path.read_bytes() for path in tmp_path.glob("*.parquet")}
    seen = []

    def fake_autorank(pivot, metric, order, hib, alpha):
        seen.append(metric)
        assert order == "ascending"
        assert not hib
        if metric != "rmse":
            assert pivot["a"].tolist() == pytest.approx([0.01] * 5)
        ranked = pd.DataFrame({"rank": [1, 2], "model": ["a", "b"], "meanrank": [1., 2.]})
        result = SimpleNamespace(alpha=alpha, effect_size="test", cd=None,
                                 pvalue=None, omnibus="test", posthoc="test")
        return ranked, result

    monkeypatch.setattr(leaderboard, "_load_write_latex_writer", lambda: None)
    monkeypatch.setattr(leaderboard, "rank_with_autorank", fake_autorank)
    monkeypatch.setattr(leaderboard, "create_report", lambda result: None)
    monkeypatch.setattr(leaderboard, "latex_table", lambda result: print("test latex"))
    monkeypatch.setattr(leaderboard, "plot_stats", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["autorank_leaderboard.py", "--output_dir", str(tmp_path)])
    leaderboard.main()

    expected = {f"absolute_marginal_coverage_error_{level}" for level in COVERAGE_LEVELS} | {
        "mean_absolute_marginal_coverage_error", "rmse",
    }
    assert set(seen) == expected
    output = capsys.readouterr().out
    dest = tmp_path / "figures" / "leaderboard"
    for metric in expected:
        data = json.loads((dest / f"cd_data_{metric}.json").read_text())
        assert data["metric"] == metric
        assert data["higher_is_better"] is False
        assert data["mean_std_rank"]["models"][0]["name"] == "a"
        assert (dest / f"cd_diagram_{metric}.png").is_file()
        assert (dest / f"latex_table_{metric}.tex").is_file()
        assert f"--- {metric} (Autorank)" in output
        assert f"--- {metric} (Mean-Std Magnitude Ranking)" in output
    assert not list(dest.glob("*_coverage_[0-9]*"))
    assert not list(dest.glob("*_amce.*"))
    for path, content in before.items():
        assert path.read_bytes() == content
