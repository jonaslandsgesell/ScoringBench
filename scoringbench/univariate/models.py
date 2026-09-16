"""Model registry for the univariate benchmark.

╔══════════════════════════════════════════════════════════════════════════╗
║  👉  REGISTER YOUR MODELS HERE  👈                                          ║
║                                                                            ║
║  Edit the ``MODELS`` dict at the bottom of this file to add / remove /     ║
║  comment out models.  Each entry maps a name to a *zero-arg factory* that returns ║
║  a fresh, unfitted wrapper, e.g.::                                         ║
║                                                                            ║
║      "my_model": lambda: MyWrapper(some_arg=123),                          ║
║                                                                            ║
║  The front script ``run_bench_regression.py`` simply does                  ║
║  ``from scoringbench.univariate.models import MODELS`` and hands it to     ║
║  ``run_benchmark``.  Nothing else needs to change.                         ║
╚══════════════════════════════════════════════════════════════════════════╝

Notes
-----
* Comment out any ``MODELS`` entry you don't want to run locally.
* Optional heavy libraries (tabpfn, tabicl, xgboost, ...) are imported lazily
  or resolve to ``None`` when missing, so this module imports cleanly even
  without every competitor stack installed.  A missing library only errors
  when its factory is actually *called* at benchmark time.
"""

from pathlib import Path

# Wrapper classes.  Missing optional libraries resolve to ``None`` here (see
# scoringbench/univariate/wrappers/__init__.py); the factories below only
# *call* these classes at fold time, so importing this module is always safe.
from .wrappers import (
    SynthefyWrapper,
    TabPFNWrapper,
    FinetuneTabPFNWrapper,
    FinetuneTabICLWrapper,
    TabICLWrapper,
    TabLDMWrapper,
    CausiloWrapper,
    TabDPTWrapper,
    XGBVectorWrapper,
    XGBQuantileVectorWrapper,
    XGBLSSWrapper,
    PytabkitRealMLPWrapper,
    PytabkitRealMLPHPOWrapper,
    PytabkitTabMDWrapper,
    PytabkitTabMHPOWrapper,
    CatBoostQuantileWrapper,
    CrepesWrapper,
    NGBoostWrapper,
    NFlowsWrapper,
    BARTWrapper,
    ForestDiffusionWrapper,
    CDEWrapper,
    FlexCodeWrapper,
    SurjectorsWrapper,
    EXAONETabularWrapper,
    MitraFinetuneWrapper,
    resolve_mitra2_checkpoint,
)
from .wrappers.cde_wrapper import CDE_PRESETS
from .wrappers.flexcode_wrapper import FLEXCODE_PRESETS
from .wrappers.surjectors_wrapper import SURJECTORS_PRESETS


def _xgb_regressor(**kwargs):
    """Construct an XGBRegressor, importing xgboost lazily so this module
    imports even when xgboost isn't installed."""
    from xgboost import XGBRegressor
    return XGBRegressor(**kwargs)


def _catboost_regressor(**kwargs):
    """Construct a CatBoostRegressor, importing catboost lazily so this module
    imports even when catboost isn't installed."""
    from catboost import CatBoostRegressor
    return CatBoostRegressor(**kwargs)


# ---------------------------------------------------------------------------
# TabPFN checkpoint paths (absolute — resolved against the project root so the
# paths work regardless of the current working directory)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH_MAP = {
    "realv2_5": str(_PROJECT_ROOT / "tabpfn-v2.5-regressor-v2.5_real.ckpt"),
    "v2_6": str(_PROJECT_ROOT / "tabpfn-v2.6-regressor-v2.6_default.ckpt"),
    "v3": str(_PROJECT_ROOT / "tabpfn-v3-regressor-v3_default.ckpt"),
}


# ---------------------------------------------------------------------------
# Finetuned-TabPFN family — one model per beta loss (built from FINETUNE_BETAS)
# ---------------------------------------------------------------------------

N_EPOCHS = 80


def _create_finetune_model_tabpfn(beta_name, tabpfn_version):
    """Factory for a finetuned-TabPFN wrapper with the given beta loss.

    Args:
        beta_name: Name of the beta loss (e.g. "crps", "crls", "wCRPS_left", "beta_0.5").
        tabpfn_version: Key into ``MODEL_PATH_MAP`` selecting the base checkpoint.

    Returns:
        A zero-arg lambda that creates a fresh ``FinetuneTabPFNWrapper``.
    """
    return lambda: FinetuneTabPFNWrapper(
        device="cuda",
        epochs=N_EPOCHS,
        learning_rate=1e-5,
        weight_decay=0.1,
        crps_loss_weight=1.0,
        mse_loss_weight=0.0,
        ce_loss_weight=0.0,
        n_finetune_ctx_plus_query_samples=20_000,
        n_estimators_finetune=1,
        n_estimators_validation=8,
        n_estimators_final_inference=8,
        early_stopping=True,
        early_stopping_patience=20,
        finetune_ctx_query_split_ratio=0.4,
        extra_regressor_kwargs={"average_before_softmax": True},
        beta=beta_name,
        model_path=MODEL_PATH_MAP[tabpfn_version],
    )


# To add more betas, simply append here — the model is added to MODELS automatically.
FINETUNE_BETAS = [
    "crls",
    "crps",
    "wCRPS_left",
    "wCRPS_center",
    "wCRPS_right",
    "ce",
    "beta_0.1",
    "beta_0.3",
    "beta_0.5",
    "beta_0.7",
    "beta_0.9",
    "beta_1.1",
    "beta_1.3",
    "beta_1.5",
    "beta_1.7",
    "beta_1.8",
    "beta_1.9",
    "is_90",
    "cde",
]

dict_finetuned_tabpfn_models = {
    f"finetune_tabpfn_realv2_5_{beta}": _create_finetune_model_tabpfn(beta, "realv2_5")
    for beta in FINETUNE_BETAS
}


# ---------------------------------------------------------------------------
# Preset-driven families (one model per registered preset name)
# ---------------------------------------------------------------------------

dict_cde_models = {
    f"cde_{_name}": (lambda n=_name: CDEWrapper(estimator=n, n_grid=200))
    for _name in CDE_PRESETS
}

dict_flexcode_models = {
    f"flexcode_{_name}": (lambda n=_name: FlexCodeWrapper(regressor=n, n_grid=200))
    for _name in FLEXCODE_PRESETS
}

dict_surjectors_models = {
    f"surjectors_{_name}": (lambda n=_name: SurjectorsWrapper(flow=n, n_grid=200))
    for _name in SURJECTORS_PRESETS
}


# ===========================================================================
# ██  MODELS — the benchmark line-up.  Add / comment out entries here.  ██
# Each value is a zero-arg factory returning a fresh, unfitted wrapper.
# ===========================================================================

MODELS = {
    "causilo_v1": lambda: CausiloWrapper(),
    "nori": lambda: SynthefyWrapper(),
    "nori_30m": lambda: SynthefyWrapper(model="nori-30m"),
    "tabpfn_realv2_5": lambda: TabPFNWrapper(model_path=MODEL_PATH_MAP["realv2_5"], ignore_pretraining_limits=True),
    "tabpfn_v2_6": lambda: TabPFNWrapper(model_path=MODEL_PATH_MAP["v2_6"], ignore_pretraining_limits=True),
    "tabpfn_v3": lambda: TabPFNWrapper(model_path=MODEL_PATH_MAP["v3"], ignore_pretraining_limits=True),
    "tabpfn_v3_5": lambda: TabPFNWrapper(model_version="v3.5", ignore_pretraining_limits=True),
    "tabpfn_v3_5_fast": lambda: TabPFNWrapper(model_version="v3.5-fast", ignore_pretraining_limits=True),
    **dict_finetuned_tabpfn_models,
    "finetune_tabpfn_realv2_5_mse": lambda: FinetuneTabPFNWrapper(
        device="cuda",
        epochs=N_EPOCHS,
        learning_rate=1e-5,
        weight_decay=0.1,
        crps_loss_weight=0.0,
        mse_loss_weight=1.0,
        ce_loss_weight=0.0,
        n_finetune_ctx_plus_query_samples=20_000,
        n_estimators_finetune=1,
        n_estimators_validation=8,
        n_estimators_final_inference=8,
        early_stopping=True,
        early_stopping_patience=20,
        finetune_ctx_query_split_ratio=0.4,
        extra_regressor_kwargs={"average_before_softmax": True},
        beta=None,
        early_stopping_metric="mse",
    ),
    "finetune_tabiclv2": lambda: FinetuneTabICLWrapper(
        epochs=N_EPOCHS,
        learning_rate=1e-5,
        n_estimators_finetune=2,
        n_estimators_validation=2,
        n_estimators_inference=8,
        early_stopping=True,
        patience=20,
        eval_metric="mse",
        random_state=0,
        verbose=True,
        # # #max_data_size=100 #only for datasets which otherwise OOM with 48GB VRAM, potentially with just 1 estimator
    ),
    "tabiclv2": lambda: TabICLWrapper(),
    "tabldm_v1": lambda: TabLDMWrapper(),
    "tabdptv1_3": lambda: TabDPTWrapper(n_ensembles=8),
    "exaonetabular": lambda: EXAONETabularWrapper(device="cuda:0"),
    "finetune_mitra2_steps_50": lambda: MitraFinetuneWrapper(
        checkpoint_dir=resolve_mitra2_checkpoint(),
        device="cuda",
        num_bag_folds=8,
        time_limit=3600,
        ft_steps=50,
        mem_usage_ratio=float("inf"),
    ),
    "mitra2": lambda: MitraFinetuneWrapper(
        checkpoint_dir=resolve_mitra2_checkpoint(),
        device="cuda",
        num_bag_folds=8,
        ft_steps=0,
        # AutoGluon's host-RAM guard estimates each Mitra child at ~8 GB and
        # raises NotEnoughMemoryError on memory-tight boxes, aborting the model
        # (raise_on_model_failure=True). Disable the check (the recipe exposes
        # no lever for it); the in-context forward pass itself fits in RAM.
        mem_usage_ratio=float("inf"),
    ),
    "crepes_tabiclv2": lambda: CrepesWrapper(
        # Use raw TabICL regressor from the tabicl package as base_model
        base_model=__import__("tabicl").TabICLRegressor(),
        n_quantiles=99,
        calibration_split=0.2,
        random_state=0,
        use_difficulty_estimator=True,
        use_mondrian_categorizer=False,
    ),
    "crepes_tabiclv2_mondrian": lambda: CrepesWrapper(
        base_model=__import__("tabicl").TabICLRegressor(),
        n_quantiles=99,
        calibration_split=0.2,
        random_state=0,
        use_difficulty_estimator=True,
        use_mondrian_categorizer=True,
        mondrian_no_bins=10,
    ),
    "xgb_vector": lambda: XGBVectorWrapper(n_bins=50, num_boost_round=100), #xgb_params={"device": "cpu"}
    "xgb_vector_quantile": lambda: XGBQuantileVectorWrapper(n_bins=50, num_boost_round=100), #xgb_params={"device": "cpu"}
    "catboost_quantile": lambda: CatBoostQuantileWrapper(n_quantiles=99, iterations=1000),
    "xgblss_Gaussian": lambda: XGBLSSWrapper(n_quantiles=100, num_boost_round=100, distribution="Gaussian"),

    "tabm_d": lambda: PytabkitTabMDWrapper(
        train_metric_name='multi_pinball(0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.47,0.49,0.51,0.53,0.55,0.57,0.59,0.61,0.63,0.65,0.67,0.69,0.71,0.73,0.75,0.77,0.79,0.81,0.83,0.85,0.87,0.89,0.91,0.93,0.95,0.97,0.99)',
        val_metric_name='multi_pinball(0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.47,0.49,0.51,0.53,0.55,0.57,0.59,0.61,0.63,0.65,0.67,0.69,0.71,0.73,0.75,0.77,0.79,0.81,0.83,0.85,0.87,0.89,0.91,0.93,0.95,0.97,0.99)',
        n_quantiles=50,
    ),
    "tabm_hpo_cv_8_tabarena": lambda: PytabkitTabMHPOWrapper(
        train_metric_name='multi_pinball(0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.47,0.49,0.51,0.53,0.55,0.57,0.59,0.61,0.63,0.65,0.67,0.69,0.71,0.73,0.75,0.77,0.79,0.81,0.83,0.85,0.87,0.89,0.91,0.93,0.95,0.97,0.99)',
        val_metric_name='multi_pinball(0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.47,0.49,0.51,0.53,0.55,0.57,0.59,0.61,0.63,0.65,0.67,0.69,0.71,0.73,0.75,0.77,0.79,0.81,0.83,0.85,0.87,0.89,0.91,0.93,0.95,0.97,0.99)',
        n_quantiles=50,
        hpo_space_name='tabarena',
        n_cv=8,
    ),
    "pytabkit_realmlp_td": lambda: PytabkitRealMLPWrapper(
        train_metric_name='multi_pinball(0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.47,0.49,0.51,0.53,0.55,0.57,0.59,0.61,0.63,0.65,0.67,0.69,0.71,0.73,0.75,0.77,0.79,0.81,0.83,0.85,0.87,0.89,0.91,0.93,0.95,0.97,0.99)',
        val_metric_name='multi_pinball(0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.47,0.49,0.51,0.53,0.55,0.57,0.59,0.61,0.63,0.65,0.67,0.69,0.71,0.73,0.75,0.77,0.79,0.81,0.83,0.85,0.87,0.89,0.91,0.93,0.95,0.97,0.99)',
        n_quantiles=50,
    ),
    "pytabkit_realmlp_hpo_cv_8_new": lambda: PytabkitRealMLPHPOWrapper(
        train_metric_name='multi_pinball(0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.47,0.49,0.51,0.53,0.55,0.57,0.59,0.61,0.63,0.65,0.67,0.69,0.71,0.73,0.75,0.77,0.79,0.81,0.83,0.85,0.87,0.89,0.91,0.93,0.95,0.97,0.99)',
        val_metric_name='multi_pinball(0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29,0.31,0.33,0.35,0.37,0.39,0.41,0.43,0.45,0.47,0.49,0.51,0.53,0.55,0.57,0.59,0.61,0.63,0.65,0.67,0.69,0.71,0.73,0.75,0.77,0.79,0.81,0.83,0.85,0.87,0.89,0.91,0.93,0.95,0.97,0.99)',
        n_quantiles=50,
        n_cv=8,
        hpo_space_name='tabarena-new',
        use_caruana_ensembling=True,
    ),


    "crepes_xgb_difficulty": lambda: CrepesWrapper(
        base_model=_xgb_regressor(n_estimators=100, random_state=0),
        n_quantiles=99,
        calibration_split=0.2,
        random_state=0,
        use_difficulty_estimator=True,
        use_mondrian_categorizer=False,
    ),
    "crepes_catboost_difficulty": lambda: CrepesWrapper(
        base_model=_catboost_regressor(iterations=100, verbose=False, random_state=0),
        n_quantiles=99,
        calibration_split=0.2,
        random_state=0,
        use_difficulty_estimator=True,
        use_mondrian_categorizer=False,
    ),
    "crepes_xgb_difficulty_mondrian": lambda: CrepesWrapper(
        base_model=_xgb_regressor(n_estimators=100, random_state=0),
        n_quantiles=99,
        calibration_split=0.2,
        random_state=0,
        use_difficulty_estimator=True,
        use_mondrian_categorizer=True,
    ),
    "crepes_catboost_difficulty_mondrian": lambda: CrepesWrapper(
        base_model=_catboost_regressor(iterations=100, verbose=False, random_state=0),
        n_quantiles=99,
        calibration_split=0.2,
        random_state=0,
        use_difficulty_estimator=True,
        use_mondrian_categorizer=True,
    ),
    "ngboost_gaussian": lambda: NGBoostWrapper(dist="normal", n_estimators=500, learning_rate=0.01, n_quantiles=99),

    "nflows_rqs": lambda: NFlowsWrapper(
        n_layers=4, hidden_features=64, num_bins=8,
        n_epochs=300, batch_size=256, lr=1e-3, n_samples=300,
    ),
    "pymc_bart": lambda: BARTWrapper(num_trees=50, draws=150, tune=200, chains=2, cores=1),
    "forest_diffusion_flow": lambda: ForestDiffusionWrapper(
        n_t=25, duplicate_K=100, diffusion_type="flow",
        n_estimators=100, max_depth=7, n_jobs=-1,
        n_samples=100, sample_chunk=10, random_state=0
    ),

    **dict_cde_models,
    **dict_flexcode_models,
    **dict_surjectors_models,
}
