"""Minimal fixed architecture transformer for the v0 assay."""
from __future__ import annotations

import torch
from torch import nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int = 32, d_model: int = 32, nhead: int = 4,
                 num_layers: int = 2, dim_ff: int = 64, max_len: int = 5,
                 n_classes: int = 8):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        # Construct each layer separately so each layer receives an independent
        # initialization draw under the model seed.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                dropout=0.0, batch_first=True, activation="gelu", norm_first=True
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pad_mask = tokens.eq(0)
        x = self.token(tokens) + self.pos[:, :tokens.shape[1]]
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad_mask)
        x = self.norm(x[:, 0])  # CLS position, never padding.
        return self.head(x)
