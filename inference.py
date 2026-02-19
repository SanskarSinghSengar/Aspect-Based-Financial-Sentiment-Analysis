#!/usr/bin/env python3
"""
hybrid_model.py

- HybridTokenDataset: wraps tokenized inputs + topic vectors + labels.
- HybridFinModel: FinBERT backbone + BiGRU + Multi-Head Attention + MLP head.
- FocalLoss: focuses training on hard examples (Neutral vs Pos/Neg confusion).
- load_tokenizer_and_backbone: loads FinBERT and controls which layers are trainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel

from .config import ARTIFACTS_DIR

MODEL_NAME = "yiyanghkust/finbert-tone"


# ---------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------
class HybridTokenDataset(Dataset):
    def __init__(self, encodings, topics: np.ndarray, labels: np.ndarray):
        self.input_ids = encodings["input_ids"]
        self.att_mask = encodings["attention_mask"]
        self.topics = torch.tensor(topics, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.att_mask[idx],
            self.topics[idx],
            self.labels[idx],
        )


# ---------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss forces the model to focus on hard-to-classify examples
    (especially Neutral vs Positive/Negative).

    gamma: focusing parameter (higher -> more focus on hard examples).
    alpha: optional class weights tensor of shape [num_classes].
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: [B, C] logits
        targets: [B] class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)  # probability of the true class
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------
# HybridFinModel with BiGRU + Multi-Head Self-Attention
# ---------------------------------------------------------------------
class HybridFinModel(nn.Module):
    def __init__(
        self,
        backbone,
        hidden_size: int,
        topics_dim: int,
        num_classes: int = 3,
        gru_hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.gru_hidden = gru_hidden

        # 1. Bi-Directional GRU (captures sequential context)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )  # output dim = gru_hidden * 2

        # 2. Multi-Head Self-Attention on top of GRU outputs
        #    We use 4 heads by default. embed_dim = gru_hidden * 2 (bidirectional)
        self.mha = nn.MultiheadAttention(
            embed_dim=gru_hidden * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # 3. Fusion head: [text_repr || topic_vec] -> MLP classifier
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden * 2 + topics_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, topics: torch.Tensor):
        """
        ids:    [B, L]
        mask:   [B, L] (1=keep, 0=pad)
        topics: [B, topics_dim]
        """
        # A. FinBERT backbone
        bert_out = self.backbone(
            input_ids=ids,
            attention_mask=mask,
            return_dict=True,
        ).last_hidden_state  # [B, L, hidden_size]

        # B. BiGRU over token sequence
        gru_out, _ = self.gru(bert_out)  # [B, L, 2*gru_hidden]

        # C. Multi-Head Self-Attention
        # key_padding_mask: True=ignore, False=keep
        key_pad_mask = (mask == 0).to(gru_out.device)  # [B, L]

        # attn_output:  [B, L, 2*gru_hidden]
        # attn_weights: [B, L, L] (each output token's dist over input tokens)
        attn_output, attn_weights_full = self.mha(
            gru_out,
            gru_out,
            gru_out,
            key_padding_mask=key_pad_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        # We want a single importance score per token for XAI.
        # Take mean over "query" positions: [B, L, L] -> [B, L]
        token_attn_scores = attn_weights_full.mean(dim=1)  # [B, L]

        # D. Mask-aware mean pooling over sequence to get one vector per example
        #    (ignoring padding tokens)
        input_mask_expanded = mask.unsqueeze(-1).expand(attn_output.size()).float()  # [B, L, 2H]
        sum_embeddings = torch.sum(attn_output * input_mask_expanded, dim=1)         # [B, 2H]
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)             # [B, 2H]
        context_vector = sum_embeddings / sum_mask                                   # [B, 2H]

        # E. Fuse with topic vector and classify
        topics = topics.to(context_vector.device)  # [B, topics_dim]
        x = torch.cat([context_vector, topics], dim=1)  # [B, 2H + topics_dim]

        logits = self.fc(x)  # [B, num_classes]
        # Return logits + 1D per-token attention scores (for inference XAI)
        return logits, token_attn_scores


# ---------------------------------------------------------------------
# Backbone loader (FinBERT)
# ---------------------------------------------------------------------
def load_tokenizer_and_backbone(
    device: torch.device,
    unfreeze_last_k: int = 4,
):
    """
    Load tokenizer + FinBERT backbone and control which encoder layers are trainable.

    During training:
      - pass unfreeze_last_k=4 (or similar) to fine-tune top layers.
    During inference:
      - pass unfreeze_last_k=0 to keep everything frozen (no gradients needed).
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME)
    backbone.to(device)

    # Freeze everything by default
    for p in backbone.parameters():
        p.requires_grad = False

    # Try to unfreeze the last K encoder layers (high-level financial semantics)
    try:
        layers = backbone.encoder.layer
        total_layers = len(layers)
        if unfreeze_last_k > 0:
            start = max(0, total_layers - unfreeze_last_k)
            for i in range(start, total_layers):
                for p in layers[i].parameters():
                    p.requires_grad = True
            # Optionally unfreeze pooler if present
            if hasattr(backbone, "pooler") and backbone.pooler is not None:
                for p in backbone.pooler.parameters():
                    p.requires_grad = True
            print(f"Unfroze last {unfreeze_last_k} encoder layers ({start}..{total_layers-1})")
        else:
            print("Backbone fully frozen for inference (unfreeze_last_k=0).")
    except Exception as e:
        # Fallback: unfreeze last ~unfreeze_last_k * 12 parameters sets
        named = list(backbone.named_parameters())
        approx = max(1, unfreeze_last_k * 12)
        for _, p in named[-approx:]:
            p.requires_grad = True
        print("Fallback unfreeze:", e)

    total = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Backbone params total={total:,} trainable={trainable:,}")

    return tokenizer, backbone
