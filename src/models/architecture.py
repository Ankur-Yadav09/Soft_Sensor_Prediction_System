"""
src/models/architecture.py
===========================
IndustrialDAE — Multi-task Denoising Autoencoder for industrial soft sensor
prediction in process industries.

Architecture
------------
  Encoder  : input_dim → 128 → 64 → latent_dim
              (BatchNorm1d, ReLU, Dropout after first linear)
  Decoder  : latent_dim → 64 → 128 → input_dim
              (reconstructs all sensor features from the latent code)
  Predictor: latent_dim → 32 → 16 → output_dim
              (predicts target KPI values from the same latent code)

Training objective
------------------
  total_loss = MSE(reconstruction) + weight_to_pred × Huber(prediction)

This module contains the model class only.  Training, persistence, and
inference utilities live in separate modules under src/training/ and
src/persistence/.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config.settings import (
    DECODER_HIDDEN_1,
    DECODER_HIDDEN_2,
    ENCODER_HIDDEN_1,
    ENCODER_HIDDEN_2,
    PREDICTOR_HIDDEN_1,
    PREDICTOR_HIDDEN_2,
)


class IndustrialDAE(nn.Module):
    """
    Denoising Autoencoder with a shared latent space for reconstruction and
    KPI prediction.

    Parameters
    ----------
    input_dim    : Number of input sensor features  (X columns)
    latent_dim   : Bottleneck dimension
    output_dim   : Number of target KPI features    (Y columns)
    dropout_rate : Dropout probability applied after the first encoder layer
    """

    def __init__(
        self,
        input_dim: int = 41,
        latent_dim: int = 15,
        output_dim: int = 5,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        # --- ENCODER: learns the "hidden physics" of the process ---
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, ENCODER_HIDDEN_1),
            nn.BatchNorm1d(ENCODER_HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ENCODER_HIDDEN_1, ENCODER_HIDDEN_2),
            nn.ReLU(),
            nn.Linear(ENCODER_HIDDEN_2, latent_dim),  # compressed state
        )

        # --- DECODER: reconstructs / "heals" all sensor features ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, DECODER_HIDDEN_1),
            nn.ReLU(),
            nn.Linear(DECODER_HIDDEN_1, DECODER_HIDDEN_2),
            nn.ReLU(),
            nn.Linear(DECODER_HIDDEN_2, input_dim),
        )

        # --- PREDICTOR: directly targets the KPI outputs ---
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, PREDICTOR_HIDDEN_1),
            nn.ReLU(),
            nn.Linear(PREDICTOR_HIDDEN_1, PREDICTOR_HIDDEN_2),
            nn.ReLU(),
            nn.Linear(PREDICTOR_HIDDEN_2, output_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Returns
        -------
        reconstructed_x : Tensor shape (batch, input_dim)
        predicted_y     : Tensor shape (batch, output_dim)
        """
        z = self.encoder(x)
        reconstructed_x = self.decoder(z)
        predicted_y = self.predictor(z)
        return reconstructed_x, predicted_y
