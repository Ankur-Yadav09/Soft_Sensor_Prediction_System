"""
src/persistence/model_store.py
================================
Model persistence utilities — save, load, and list IndustrialDAE artifacts.

Each model is stored in its own sub-directory under MODEL_DIR:

    saved_models/<model_name>/
        model.pth        PyTorch checkpoint
                         keys: state_dict, input_dim, latent_dim, output_dim
        scaler_x.pkl     fitted StandardScaler for X features
        scaler_y.pkl     fitted StandardScaler for Y targets
        columns.pkl      {'x_cols': [...], 'y_cols': [...]}
        metadata.pkl     human-readable info dict (name, saved_at, dims, cols)

Public API
----------
save_model_to_disk(model, scaler_x, scaler_y, x_cols, y_cols, model_name)
load_model_from_disk(model_name) → (model, scaler_x, scaler_y, x_cols, y_cols)
list_saved_models()              → list[dict]
"""
from __future__ import annotations

import datetime
import os
import pickle
from typing import List, Tuple

import torch

from config.settings import MODEL_DIR
from src.models.architecture import IndustrialDAE


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_model_to_disk(
    model: IndustrialDAE,
    scaler_x,
    scaler_y,
    x_cols: List[str],
    y_cols: List[str],
    model_name: str,
) -> None:
    """
    Persist a trained model with its scalers, column config, and metadata.

    The function is idempotent: calling it a second time with the same
    model_name will silently overwrite the previous artifacts.
    """
    save_path = os.path.join(MODEL_DIR, model_name)
    os.makedirs(save_path, exist_ok=True)

    # --- PyTorch checkpoint ---
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim":  len(x_cols),
            "latent_dim": model.encoder[-1].out_features,
            "output_dim": model.predictor[-1].out_features,
        },
        os.path.join(save_path, "model.pth"),
    )

    # --- Scalers ---
    for fname, obj in [("scaler_x.pkl", scaler_x), ("scaler_y.pkl", scaler_y)]:
        with open(os.path.join(save_path, fname), "wb") as fh:
            pickle.dump(obj, fh)

    # --- Column config ---
    with open(os.path.join(save_path, "columns.pkl"), "wb") as fh:
        pickle.dump({"x_cols": x_cols, "y_cols": y_cols}, fh)

    # --- Human-readable metadata ---
    meta = {
        "name":       model_name,
        "saved_at":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_dim":  len(x_cols),
        "output_dim": len(y_cols),
        "x_cols":     x_cols,
        "y_cols":     y_cols,
    }
    with open(os.path.join(save_path, "metadata.pkl"), "wb") as fh:
        pickle.dump(meta, fh)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_model_from_disk(
    model_name: str,
) -> Tuple[IndustrialDAE, object, object, List[str], List[str]]:
    """
    Reconstruct a saved model from disk artifacts.

    Parameters
    ----------
    model_name : sub-directory name inside MODEL_DIR

    Returns
    -------
    model, scaler_x, scaler_y, x_cols, y_cols
    """
    load_path = os.path.join(MODEL_DIR, model_name)

    # Reconstruct architecture from saved dims, then load weights
    ckpt = torch.load(
        os.path.join(load_path, "model.pth"), weights_only=False
    )
    model = IndustrialDAE(
        input_dim=ckpt["input_dim"],
        latent_dim=ckpt["latent_dim"],
        output_dim=ckpt["output_dim"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with open(os.path.join(load_path, "scaler_x.pkl"), "rb") as fh:
        scaler_x = pickle.load(fh)
    with open(os.path.join(load_path, "scaler_y.pkl"), "rb") as fh:
        scaler_y = pickle.load(fh)
    with open(os.path.join(load_path, "columns.pkl"), "rb") as fh:
        cols = pickle.load(fh)

    return model, scaler_x, scaler_y, cols["x_cols"], cols["y_cols"]


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


def list_saved_models() -> List[dict]:
    """
    Return a list of metadata dicts for every model saved to disk.

    Models without a readable metadata.pkl are silently skipped.
    """
    models: List[dict] = []
    if not os.path.exists(MODEL_DIR):
        return models

    for name in os.listdir(MODEL_DIR):
        meta_path = os.path.join(MODEL_DIR, name, "metadata.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as fh:
                models.append(pickle.load(fh))

    return models
