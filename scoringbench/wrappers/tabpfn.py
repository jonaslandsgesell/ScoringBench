"""TabPFN wrapper for ScoringBench."""

from __future__ import annotations

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper


import sys
import os

# # Get the absolute path to the current directory
project_root = "/home/landsges/ScoringBench"
# Insert it at the beginning of the search path
sys.path.insert(0, project_root) ## so that modified tabpfn is preferred


class TabPFNWrapper(ProbabilisticWrapper):
    """Wraps TabPFNRegressor with a DistributionPrediction interface.

    The predictive distribution is read directly from TabPFN's native
    bar-distribution output (logits + borders) via output_type='full'.
    """

    def __init__(self, device=None, **kwargs):
        import torch
        from tabpfn import TabPFNRegressor
        kwargs.pop('device', None)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model = TabPFNRegressor(device=self._device, **kwargs)

    def fit(self, X, y) -> "TabPFNWrapper":
        self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self._model.predict(X))

    def predict_distribution(self, X) -> DistributionPrediction:
        import torch
        with torch.no_grad():
            pred_full = self._model.predict(X, output_type="full")

        logits = pred_full["logits"]
        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits, device=self._device)
        else:
            logits = logits.to(self._device)

        criterion = pred_full["criterion"]
        bin_edges = criterion.borders.cpu().numpy()             # (n_bins+1,)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2   # (n_bins,)

        probas = torch.softmax(logits, dim=-1).cpu().numpy()    # (n_samples, n_bins)
        mean = (probas * bin_midpoints[None, :]).sum(axis=-1)   # (n_samples,)

        return DistributionPrediction(
            probas=probas,
            bin_edges=bin_edges,
            bin_midpoints=bin_midpoints,
            mean=mean,
        )


class FinetuneTabPFNWrapper(ProbabilisticWrapper):
    """Wraps FinetunedTabPFNRegressor with the same DistributionPrediction interface
    as TabPFNWrapper.  All finetuning hyperparameters can be passed directly.

    Parameters
    ----------
    device : str
        Device for training/inference, e.g. "cuda" or "cpu".
    epochs : int
        Total finetuning epochs.
    time_limit : int | None
        Optional wall-clock time limit in seconds.
    learning_rate : float
        AdamW learning rate.
    weight_decay : float
        AdamW weight decay.
    validation_split_ratio : float
        Fraction of training data held out for early-stopping.
    n_finetune_ctx_plus_query_samples : int
        Total samples per meta-dataset during finetuning.
    finetune_ctx_query_split_ratio : float
        Fraction of each meta-dataset used as query samples.
    n_inference_subsample_samples : int
        Subsampled training samples per estimator at inference time.
    random_state : int
        Seed for reproducibility.
    early_stopping : bool
        Enable early stopping.
    early_stopping_patience : int
        Patience epochs for early stopping.
    min_delta : float
        Minimum improvement to count as progress.
    grad_clip_value : float | None
        Max gradient norm; None disables clipping.
    use_lr_scheduler : bool
        Use linear-warmup (+ optional cosine decay) scheduler.
    lr_warmup_only : bool
        If True, only apply linear warmup and hold the LR constant afterwards.
    n_estimators_finetune : int
        Ensemble size during finetuning.
    n_estimators_validation : int
        Ensemble size during validation.
    n_estimators_final_inference : int
        Ensemble size for final inference.
    use_activation_checkpointing : bool
        Reduces GPU memory at the cost of extra compute.
    save_checkpoint_interval : int | None
        How often (in epochs) to save intermediate checkpoints.
    extra_regressor_kwargs : dict | None
        Extra kwargs forwarded to the underlying TabPFNRegressor.
    ce_loss_weight : float
        Weight for bar-distribution cross-entropy loss.
    crps_loss_weight : float
        Weight for CRPS loss.
    crls_loss_weight : float
        Weight for CRLS (log-score) loss.
    mse_loss_weight : float
        Weight for auxiliary MSE loss.
    mse_loss_clip : float | None
        Optional upper bound for the MSE loss term.
    mae_loss_weight : float
        Weight for auxiliary MAE loss.
    mae_loss_clip : float | None
        Optional upper bound for the MAE loss term.
    beta : str | float
        Scoring rule used for both the finetuning loss and early-stopping
        evaluation. Supported values: ``"crls"``, ``"crps"``, ``"brier"``,
        ``"wCRPS_left"``, ``"wCRPS_center"``, ``"wCRPS_right"``,
        ``"integrated"``, ``"ce"``, ``"mse"``, or a float / ``"beta_<v>"``
        for the energy score with exponent *v*. Defaults to ``"crls"``.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        epochs: int = 30,
        time_limit: int | None = None,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        validation_split_ratio: float = 0.1,
        n_finetune_ctx_plus_query_samples: int = 10_000,
        finetune_ctx_query_split_ratio: float = 0.2,
        n_inference_subsample_samples: int = 50_000,
        random_state: int = 0,
        early_stopping: bool = True,
        early_stopping_patience: int = 8,
        min_delta: float = 1e-4,
        grad_clip_value: float | None = 1.0,
        use_lr_scheduler: bool = True,
        lr_warmup_only: bool = False,
        n_estimators_finetune: int = 2,
        n_estimators_validation: int = 2,
        n_estimators_final_inference: int = 8,
        use_activation_checkpointing: bool = True,
        save_checkpoint_interval: int | None = 10,
        extra_regressor_kwargs: dict | None = None,
        ce_loss_weight: float = 0.0,
        crps_loss_weight: float = 1.0,
        crls_loss_weight: float = 0.0,
        mse_loss_weight: float = 1.0,
        mse_loss_clip: float | None = None,
        mae_loss_weight: float = 0.0,
        mae_loss_clip: float | None = None,
        early_stopping_metric: str | None = None,  # None defaults to "same_as_beta"
        beta: str | float | None = None,
        **kwargs
    ):
        import torch
        from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model = FinetunedTabPFNRegressor(
            device=device,
            epochs=epochs,
            time_limit=time_limit,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            validation_split_ratio=validation_split_ratio,
            n_finetune_ctx_plus_query_samples=n_finetune_ctx_plus_query_samples,
            finetune_ctx_query_split_ratio=finetune_ctx_query_split_ratio,
            n_inference_subsample_samples=n_inference_subsample_samples,
            random_state=random_state,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            min_delta=min_delta,
            grad_clip_value=grad_clip_value,
            use_lr_scheduler=use_lr_scheduler,
            lr_warmup_only=lr_warmup_only,
            n_estimators_finetune=n_estimators_finetune,
            n_estimators_validation=n_estimators_validation,
            n_estimators_final_inference=n_estimators_final_inference,
            use_activation_checkpointing=use_activation_checkpointing,
            save_checkpoint_interval=save_checkpoint_interval,
            extra_regressor_kwargs=extra_regressor_kwargs,
            ce_loss_weight=ce_loss_weight,
            crps_loss_weight=crps_loss_weight,
            crls_loss_weight=crls_loss_weight,
            mse_loss_weight=mse_loss_weight,
            mse_loss_clip=mse_loss_clip,
            mae_loss_weight=mae_loss_weight,
            mae_loss_clip=mae_loss_clip,
            early_stopping_metric=early_stopping_metric,
            beta=beta,
            **kwargs
        )

    def fit(self, X, y) -> "FinetuneTabPFNWrapper":
        self._model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self._model.predict(X))

    def predict_distribution(self, X) -> DistributionPrediction:
        import torch
        with torch.no_grad():
            pred_full = self._model.predict(X, output_type="full")

        logits = pred_full["logits"]
        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits, device=self._device)
        else:
            logits = logits.to(self._device)
        criterion = pred_full["criterion"]
        bin_edges = criterion.borders.cpu().numpy()             # (n_bins+1,)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2   # (n_bins,)

        probas = torch.softmax(logits, dim=-1).cpu().numpy()    # (n_samples, n_bins)
        mean = (probas * bin_midpoints[None, :]).sum(axis=-1)   # (n_samples,)

        return DistributionPrediction(
            probas=probas,
            bin_edges=bin_edges,
            bin_midpoints=bin_midpoints,
            mean=mean,
        )