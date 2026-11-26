"""Categorical features: per-fold mode imputation and native pass-through.

``cv.run_fold`` hands a wrapper either the pandas ``category`` columns
(``supports_categorical = True``) or all-numeric integer codes (default).  These
toy tests pin, per wrapper, that the flag is set as audited AND -- where the
backing library is installed -- that the library really detects the toy
``category`` column as categorical.  Each test is a few-row fit (or just the
library's own type-detection function) so the whole file runs in seconds;
libraries that are not installed are skipped.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from scoringbench.univariate import cv
from scoringbench.univariate.wrappers import ProbabilisticWrapper

LEVELS = ["a", "b", "c", "d", "e"]  # >3 levels: integer codes would look numeric


def _toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    cat = pd.Categorical(rng.choice(LEVELS, n), categories=LEVELS)
    num = rng.normal(size=n)
    effect = pd.Series(cat).map(dict(zip(LEVELS, [0.0, 5.0, -3.0, 2.0, -1.0]))).to_numpy()
    X = pd.DataFrame({"num": num, "cat": cat})
    y = pd.Series(num + effect + 0.1 * rng.normal(size=n))
    return X, y


# ---------------------------------------------------------------------------
# Flag audit: which wrappers receive the category dtype
# ---------------------------------------------------------------------------

# (module, class) -> expected flag.  False = the wrapper casts X to a float
# array / numpy itself, or the library only ordinal-encodes (same as codes).
EXPECTED_FLAGS = {
    ("catboost_wrapper", "CatBoostQuantileWrapper"): True,
    ("tabpfn", "TabPFNWrapper"): True,
    ("lightpfn", "LightPFNWrapper"): True,
    ("causilo", "CausiloWrapper"): True,
    ("ashlar_dressed_wrapper", "AshlarDressedWrapper"): True,
    ("pytabkit", "PytabkitRealMLPWrapper"): True,
    ("pytabkit", "PytabkitRealMLPHPOWrapper"): True,
    ("pytabkit", "PytabkitTabMDWrapper"): True,
    ("kumo", "KumoTabularWrapper"): True,
    ("kumo", "KumoTabularSmallWrapper"): True,
    ("tabpfn", "FinetuneTabPFNWrapper"): False,
    ("tabicl", "TabICLWrapper"): False,
    ("tabicl", "FinetuneTabICLWrapper"): False,
    ("tabldm", "TabLDMWrapper"): False,
    ("tabdpt", "TabDPTWrapper"): False,
    ("synthefy", "SynthefyWrapper"): False,
    ("mitra_finetune_wrapper", "MitraFinetuneWrapper"): False,
    ("limix", "LimiXWrapper"): False,
    ("exaonetabular_wrapper", "EXAONETabularWrapper"): False,
    ("xgb_vector", "XGBVectorWrapper"): False,
    ("xgb_vector", "XGBQuantileVectorWrapper"): False,
    ("xgblss_wrapper", "XGBLSSWrapper"): True,
    ("crepes_wrapper", "CrepesWrapper"): False,
    ("ngboost_wrapper", "NGBoostWrapper"): False,
    ("bart_wrapper", "BARTWrapper"): False,
    ("cde_wrapper", "CDEWrapper"): False,
    ("flexcode_wrapper", "FlexCodeWrapper"): False,
    ("nflows_wrapper", "NFlowsWrapper"): False,
    ("forest_diffusion_wrapper", "ForestDiffusionWrapper"): False,
    ("condensite_pit_boosting_wrapper", "CondensitePITBoostingWrapper"): False,
    ("surjectors_wrapper", "SurjectorsWrapper"): False,
}


@pytest.mark.parametrize("mod,cls", list(EXPECTED_FLAGS), ids=lambda v: str(v))
def test_supports_categorical_flag(mod, cls):
    try:
        m = importlib.import_module(f"scoringbench.univariate.wrappers.{mod}")
    except Exception as exc:  # module-level import of a missing backend
        pytest.skip(f"{mod} not importable: {exc}")
    assert getattr(m, cls).supports_categorical is EXPECTED_FLAGS[(mod, cls)]


# ---------------------------------------------------------------------------
# run_fold: imputation + dtype routing (no backend needed)
# ---------------------------------------------------------------------------

def _recorder(native):
    seen = {}

    class W(ProbabilisticWrapper):
        supports_categorical = native

        def fit(self, X, y):
            seen["fit"] = X.copy()
            return self

        def predict(self, X):
            seen["pred"] = X.copy()
            return np.zeros(len(X))

    return W, seen


@pytest.mark.parametrize("native", [False, True])
def test_run_fold_imputes_categorical_mode_and_routes_dtype(native):
    X = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        "cat": pd.Categorical(["a", "b", None, "a", "c", None], categories=["a", "b", "c"]),
    })
    y = pd.Series(np.arange(6.0))
    W, seen = _recorder(native)
    cv.run_fold(X.iloc[:4], X.iloc[4:], y.iloc[:4], y.iloc[4:], {"w": W}, seed=0)

    fit, pred = seen["fit"], seen["pred"]
    if native:
        assert isinstance(fit["cat"].dtype, pd.CategoricalDtype)
        assert list(fit["cat"]) == ["a", "b", "a", "a"]   # NaN -> train mode "a"
        assert list(pred["cat"]) == ["c", "a"]           # test NaN -> TRAIN mode
    else:
        assert fit.select_dtypes(exclude=[np.number]).empty
        assert list(fit["cat"]) == [0, 1, 0, 0]
        assert list(pred["cat"]) == [2, 0]               # shared full-data codes
    assert X["cat"].isna().sum() == 2                    # caller's frame untouched


# ---------------------------------------------------------------------------
# Per-library: the toy category column is detected as categorical
# ---------------------------------------------------------------------------

def test_catboost_uses_cat_features():
    pytest.importorskip("catboost")
    from scoringbench.univariate.wrappers.catboost_wrapper import CatBoostQuantileWrapper

    X, y = _toy()
    X["cat"] = X["cat"].cat.rename_categories([1.5, 2.5, 3.5, 4.5, 5.5])  # float levels
    w = CatBoostQuantileWrapper(n_quantiles=9, iterations=10).fit(X, y)
    assert w._model.get_cat_feature_indices() == [1]
    assert np.all(np.isfinite(w.predict(X)))


def test_tabpfn_receives_categorical_indices():
    pytest.importorskip("tabpfn")
    from tabpfn.preprocessing.datamodel import FeatureModality
    from scoringbench.univariate.wrappers.tabpfn import TabPFNWrapper

    X, y = _toy()
    try:
        w = TabPFNWrapper(device="cpu", n_estimators=1).fit(X, y)
    except Exception as exc:  # weights not downloadable offline
        pytest.skip(f"TabPFN fit unavailable: {exc}")
    assert w._model.categorical_features_indices == [1]
    schema = w._model.inferred_feature_schema_
    assert 1 in schema.indices_for(FeatureModality.CATEGORICAL)
    assert np.all(np.isfinite(w.predict(X)))


def test_lightpfn_auto_detects_category_dtype():
    pytest.importorskip("lightpfn")
    from lightpfn.sklearn import auto_categorical

    X, _ = _toy()
    X["num"] = np.linspace(0, 1, len(X))  # clearly continuous
    assert auto_categorical(X).tolist() == [False, True]


def _causilo_model_version():
    sdk = pytest.importorskip("causilo")
    versions = {"1.0.1": "1.0", "1.5.1": "1.5"}
    if sdk.__version__ not in versions:
        pytest.skip("install the SDK version pinned for the Causilo model")
    return versions[sdk.__version__]


def test_causilo_encoder_detects_category_dtype():
    model_version = _causilo_model_version()
    X, _ = _toy()
    if model_version == "1.0":
        from causilo.data.encoding import FeatureEncoder

        schema, _ = FeatureEncoder.fit(X)
        assert tuple(schema.categories) == (1,)
    else:
        from causilo.v1_5.data.encoding import InputLayout

        assert InputLayout.from_table(X).categorical_columns == (1,)


def test_ashlar_splits_off_category_columns():
    pytest.importorskip("ashlar")
    from ashlar.additive import _frame_parts

    X, _ = _toy()
    numeric, categoricals = _frame_parts(X)
    assert numeric.shape == (len(X), 1) and len(categoricals) == 1


def test_kumo_infers_category_as_categorical_but_codes_as_numerical():
    sdm = pytest.importorskip("sdm")

    X, _ = _toy()
    assert str(sdm.infer_stypes(X)["cat"]).endswith("categorical")
    codes = X.assign(cat=X["cat"].cat.codes)
    assert str(sdm.infer_stypes(codes)["cat"]).endswith("numerical")


def test_pytabkit_realmlp_fits_category_frame():
    pytest.importorskip("pytabkit")
    from scoringbench.univariate.wrappers.pytabkit import PytabkitRealMLPWrapper

    X, y = _toy()
    try:
        w = PytabkitRealMLPWrapper(n_epochs=2, device="cpu").fit(X, y)
    except TypeError:
        w = PytabkitRealMLPWrapper().fit(X, y)
    assert np.all(np.isfinite(w.predict(X)))


def test_xgblss_uses_native_categorical_splits():
    pytest.importorskip("xgboostlss")
    from scoringbench.univariate.wrappers.xgblss_wrapper import XGBLSSWrapper

    X, y = _toy()
    w = XGBLSSWrapper(n_quantiles=9, num_boost_round=5).fit(X, y)
    assert w._model.booster.feature_types == ["float", "c"]
    assert np.all(np.isfinite(w.predict(X)))


# ---------------------------------------------------------------------------
# Crepes: base learner sees categories, difficulty estimator / Mondrian see codes
# ---------------------------------------------------------------------------

def _crepes_fit(base, mondrian=False):
    pytest.importorskip("crepes")
    from scoringbench.univariate.wrappers.crepes_wrapper import CrepesWrapper

    X, y = _toy(n=120)
    w = CrepesWrapper(base_model=base, random_state=0, native_categorical=True,
                      use_mondrian_categorizer=mondrian, mondrian_no_bins=3).fit(X, y)
    assert w.supports_categorical
    assert np.all(np.isfinite(w.predict(X.iloc[:10])))
    return w


@pytest.mark.parametrize("mondrian", [False, True])
def test_crepes_xgb_base_gets_native_categorical_splits(mondrian):
    xgb = pytest.importorskip("xgboost")
    base = xgb.XGBRegressor(n_estimators=5, enable_categorical=True)
    _crepes_fit(base, mondrian)
    assert base.get_booster().feature_types == ["float", "c"]


@pytest.mark.parametrize("mondrian", [False, True])
def test_crepes_catboost_base_gets_cat_features(mondrian):
    catboost = pytest.importorskip("catboost")
    base = catboost.CatBoostRegressor(iterations=5, verbose=False)
    _crepes_fit(base, mondrian)
    assert base.get_cat_feature_indices() == [1]


def test_only_xgb_catboost_crepes_models_opt_in():
    pytest.importorskip("crepes")
    from scoringbench.univariate.models import MODELS

    crepes = {k: f for k, f in MODELS.items() if k.startswith("crepes_")}
    for k, f in crepes.items():
        if "tabicl" in k:
            continue  # skip building TabICL; it stays on codes by default
        expected = ("xgb" in k) or ("catboost" in k)
        assert f().supports_categorical is expected, k


# ---------------------------------------------------------------------------
# Unseen test-fold levels: categories are fixed on the full dataset (shared
# dtype across folds), so a level can be present in test but absent in train.
# Native-categorical wrappers must still predict finitely.
# ---------------------------------------------------------------------------

def _unseen_factories():
    out = {}

    def add(name, mod, build):
        try:
            importlib.import_module(mod)
            out[name] = build
        except Exception:
            pass

    add("catboost", "catboost", lambda: __import__(
        "scoringbench.univariate.wrappers.catboost_wrapper", fromlist=["x"]
    ).CatBoostQuantileWrapper(n_quantiles=9, iterations=10))
    add("tabpfn", "tabpfn", lambda: __import__(
        "scoringbench.univariate.wrappers.tabpfn", fromlist=["x"]
    ).TabPFNWrapper(n_estimators=1))
    add("xgblss", "xgboostlss", lambda: __import__(
        "scoringbench.univariate.wrappers.xgblss_wrapper", fromlist=["x"]
    ).XGBLSSWrapper(n_quantiles=9, num_boost_round=5))
    add("lightpfn", "lightpfn", lambda: __import__(
        "scoringbench.univariate.wrappers.lightpfn", fromlist=["x"]
    ).LightPFNWrapper())
    add("causilo", "causilo", lambda: __import__(
        "scoringbench.univariate.wrappers.causilo", fromlist=["x"]
    ).CausiloWrapper(n_estimators=1, model_version=_causilo_model_version()))
    add("kumo_small", "sdm", lambda: __import__(
        "scoringbench.univariate.wrappers.kumo", fromlist=["x"]
    ).KumoTabularSmallWrapper())

    def crepes(base):
        from scoringbench.univariate.wrappers.crepes_wrapper import CrepesWrapper
        return CrepesWrapper(base_model=base(), random_state=0, native_categorical=True)

    if importlib.util.find_spec("crepes"):
        add("crepes_xgb", "xgboost", lambda: crepes(
            lambda: __import__("xgboost").XGBRegressor(n_estimators=5, enable_categorical=True)))
        add("crepes_catboost", "catboost", lambda: crepes(
            lambda: __import__("catboost").CatBoostRegressor(iterations=5, verbose=False)))
    return out


_UNSEEN = _unseen_factories()


@pytest.mark.parametrize("name", sorted(_UNSEEN) or ["none"])
def test_native_wrappers_predict_on_level_unseen_in_train(name):
    if name == "none":
        pytest.skip("no native-categorical backend installed")
    X, y = _toy(n=80)
    unseen = X["cat"] == "e"
    X_tr, y_tr = X[~unseen], y[~unseen]
    X_te = X[unseen]
    assert len(X_te) > 0 and "e" not in set(X_tr["cat"])
    assert X_tr["cat"].dtype == X_te["cat"].dtype  # shared full-dataset levels

    model = _UNSEEN[name]()
    assert model.supports_categorical
    model.fit(X_tr, y_tr)
    pred = np.asarray(model.predict(X_te), dtype=float)
    assert pred.shape == (len(X_te),) and np.all(np.isfinite(pred))
    dist = model.predict_distribution(X_te)
    assert np.all(np.isfinite(np.asarray(dist.mean, dtype=float)))
