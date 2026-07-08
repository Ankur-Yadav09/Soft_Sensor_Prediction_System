from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel

ALGORITHMS = ("DAE", "Random Forest", "XGBoost", "LightGBM", "LSTM", "Kalman Filter")


class TrainingRequest(BaseModel):
    project_id: str
    algorithm: str
    hyperparameters: Dict[str, Any] = {}
