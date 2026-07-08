"""
backend/app/services/training_service.py
============================================
Dispatches to the 4 existing trainer functions by algorithm name, loading
their inputs fresh from a persisted project_id (see project_service) rather
than any in-memory state. Each algorithm gets its own thin closure so the
job manager can bind the progress-callback shape that function actually
accepts (see backend/app/jobs/manager.py's ProgressMode):

    DAE, LSTM              -> "epoch"   (progress_callback(current,total) + status_callback)
    Random Forest/XGBoost/
    LightGBM, Kalman Filter -> "none"   (train_sklearn_model/train_kalman_model absorb
                                          arbitrary kwargs via **hparams; a stray
                                          callback kwarg would crash the estimator ctor)

On completion each closure persists the trained model exactly as train.py's
UI does today (save_model_to_disk + save_model_to_registry, unchanged) and
returns a small JSON-safe summary — never the raw wrapper object — so the
job manager only ever holds serializable data.
"""
from __future__ import annotations

import datetime
import os

from config.settings import MODEL_DIR
from src.data.database import save_model_to_registry
from src.evaluation.metrics import compute_metrics
from src.models.wrappers import DAEWrapper
from src.persistence.model_store import save_model_to_disk
from src.training.train_kalman import train_kalman_model
from src.training.train_lstm import train_lstm
from src.training.train_sklearn import train_sklearn_model
from src.training.trainer import train_model

from backend.app.services import project_service

ALGORITHMS = ("DAE", "Random Forest", "XGBoost", "LightGBM", "LSTM", "Kalman Filter")
_SKLEARN_ALGORITHMS = {"Random Forest", "XGBoost", "LightGBM"}


def _finish(project: project_service.ProjectArtifacts, algorithm: str, wrapper, loss_history: dict) -> dict:
    preds = project.scaler_y.inverse_transform(wrapper.predict_scaled(project.X_test))  # unchanged
    metrics_df = compute_metrics(project.y_test_raw, preds, project.y_cols)  # unchanged
    avg_r2 = float(metrics_df["R2 Score"].mean())
    avg_rmse = float(metrics_df["RMSE"].mean())
    avg_mae = float(metrics_df["MAE"].mean())

    model_name = (
        f"{algorithm.replace(' ', '_')}_{project.project_id}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    save_model_to_disk(  # unchanged
        wrapper, project.scaler_x, project.scaler_y, project.x_cols, project.y_cols, model_name
    )
    try:
        save_model_to_registry(  # unchanged
            model_name=model_name,
            algorithm=algorithm,
            dataset_name=project.dataset_name,
            x_cols=project.x_cols,
            y_cols=project.y_cols,
            avg_r2=avg_r2,
            avg_rmse=avg_rmse,
            avg_mae=avg_mae,
            file_path=os.path.join(MODEL_DIR, model_name),
        )
    except Exception:
        pass  # registry write failure must not block training success (matches train.py today)

    return {
        "model_name": model_name,
        "algorithm": algorithm,
        "avg_r2": avg_r2,
        "avg_rmse": avg_rmse,
        "avg_mae": avg_mae,
        "actual_epochs": loss_history.get("actual_epochs"),
        "early_stopped": loss_history.get("early_stopped", False),
        # Present (non-empty) only for DAE/LSTM — tree ensembles and Kalman
        # Filter return empty lists here (see train_sklearn.py/train_kalman.py),
        # which the frontend uses to decide whether to render loss curves.
        "epoch_recon_losses": loss_history.get("epoch_recon_losses", []),
        "epoch_pred_losses": loss_history.get("epoch_pred_losses", []),
        "val_recon_losses": loss_history.get("val_recon_losses", []),
        "val_pred_losses": loss_history.get("val_pred_losses", []),
    }


def prepare_training_job(project_id: str, algorithm: str, hyperparameters: dict):
    """
    Eagerly loads the project (so a bad project_id 404s immediately from the
    route rather than surfacing only via job polling) and returns the target
    closure + progress_mode the route should submit to the job manager.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Expected one of {ALGORITHMS}.")

    project = project_service.load_project(project_id)  # unchanged, 404s via HTTPException

    if algorithm == "DAE":
        def target(progress_callback=None, status_callback=None):
            dae, loss_history = train_model(  # unchanged
                X_train=project.X_train,
                y_train=project.y_train,
                X_test=project.X_test,
                y_test_scaled=project.y_test,
                y_test_raw=project.y_test_raw,
                y_cols=project.y_cols,
                scaler_y=project.scaler_y,
                progress_callback=progress_callback,
                status_callback=status_callback,
                **hyperparameters,
            )
            wrapper = DAEWrapper(dae)
            return _finish(project, algorithm, wrapper, loss_history)

        return target, "epoch"

    if algorithm == "LSTM":
        def target(progress_callback=None, status_callback=None):
            wrapper, loss_history = train_lstm(  # unchanged
                X_train=project.X_train,
                y_train_scaled=project.y_train,
                X_test=project.X_test,
                y_test_scaled=project.y_test,
                y_test_raw=project.y_test_raw,
                y_cols=project.y_cols,
                scaler_y=project.scaler_y,
                progress_callback=progress_callback,
                status_callback=status_callback,
                **hyperparameters,
            )
            return _finish(project, algorithm, wrapper, loss_history)

        return target, "epoch"

    if algorithm in _SKLEARN_ALGORITHMS:
        def target():
            wrapper, loss_history = train_sklearn_model(  # unchanged
                X_train=project.X_train,
                y_train_scaled=project.y_train,
                X_test=project.X_test,
                y_test_scaled=project.y_test,
                y_test_raw=project.y_test_raw,
                y_cols=project.y_cols,
                scaler_y=project.scaler_y,
                model_type=algorithm,
                **hyperparameters,
            )
            return _finish(project, algorithm, wrapper, loss_history)

        return target, "none"

    # Kalman Filter
    def target():
        wrapper, loss_history = train_kalman_model(  # unchanged
            X_train=project.X_train,
            y_train_scaled=project.y_train,
            X_test=project.X_test,
            y_test_scaled=project.y_test,
            y_test_raw=project.y_test_raw,
            y_cols=project.y_cols,
            scaler_y=project.scaler_y,
            **hyperparameters,
        )
        return _finish(project, algorithm, wrapper, loss_history)

    return target, "none"
