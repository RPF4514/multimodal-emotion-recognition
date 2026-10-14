"""Minimal bert-base-uncased encoder implemented in plain PyTorch.

Why hand-rolled: the training environment has torch but not ``transformers``,
and the competition only needs a frozen feature-extraction forward pass. This
module loads a HuggingFace ``pytorch_model.bin`` state dict directly, so no
extra dependency is required.

Interface contract (identical to the supplied attachments):
    text_bert : int tensor [B, 3, 50]  ->  (token_ids, attention_mask, token_type_ids)
    returns   : float tensor [B, 50, 768]

The competition supplies ``text_bert`` in attachments 2/3/4 (the aligned
versions all carry it), so this one encoder serves every attachment and keeps
the "same feature version, same input interface" requirement intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class BertConfig:
    def __init__(self, raw: dict):
        self.vocab_size = int(raw["vocab_size"])
        self.hidden_size = int(raw["hidden_size"])
        self.num_hidden_layers = int(raw["num_hidden_layers"])
        self.num_attention_heads = int(raw["num_attention_heads"])
        self.intermediate_size = int(raw["intermediate_size"])
        self.max_position_embeddings = int(raw["max_position_embeddings"])
        self.type_vocab_size = int(raw.get("type_vocab_size", 2))
        self.layer_norm_eps = float(raw.get("layer_norm_eps", 1e-12))
        self.hidden_act = raw.get("hidden_act", "gelu")


class _SelfAttention(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.key = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, hidden: torch.Tensor, additive_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = hidden.shape
        shape = (bsz, seq_len, self.num_heads, self.head_dim)

        def split(x):
            return x.view(*shape).transpose(1, 2)

        query = split(self.query(hidden))
        key = split(self.key(hidden))
        value = split(self.value(hidden))
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        scores = scores + additive_mask
        probs = F.softmax(scores, dim=-1)
        context = torch.matmul(probs, value)
        context = context.transpose(1, 2).reshape(bsz, seq_len, dim)
        return context


class _Layer(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = _SelfAttention(cfg)
        self.attention.output = nn.Module()
        self.attention.output.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.attention.output.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.intermediate = nn.Module()
        self.intermediate.dense = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.output = nn.Module()
        self.output.dense = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        self.output.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)


class BertEncoder(nn.Module):
    """Frozen bert-base-uncased encoder returning ``last_hidden_state``."""

    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.cfg = cfg
        self.embeddings = nn.Module()
        self.embeddings.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.embeddings.position_embeddings = nn.Embedding(cfg.max_position_embeddings,
                                                           cfg.hidden_size)
        self.embeddings.token_type_embeddings = nn.Embedding(cfg.type_vocab_size,
                                                             cfg.hidden_size)
        self.embeddings.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([_Layer(cfg) for _ in range(cfg.num_hidden_layers)])

    def forward(self, text_bert: torch.Tensor) -> torch.Tensor:
        if text_bert.dtype not in (torch.int64, torch.int32):
            text_bert = text_bert.long()
        token_ids = text_bert[:, 0, :]
        attention_mask = text_bert[:, 1, :]
        token_types = text_bert[:, 2, :] if text_bert.shape[1] > 2 else torch.zeros_like(token_ids)
        bsz, seq_len = token_ids.shape

        positions = torch.arange(seq_len, device=token_ids.device)[None, :].expand(bsz, seq_len)
        hidden = (self.embeddings.word_embeddings(token_ids)
                  + self.embeddings.position_embeddings(positions)
                  + self.embeddings.token_type_embeddings(token_types))
        hidden = self.embeddings.LayerNorm(hidden)

        # Additive mask: 0 where attended, large negative where padded.
        additive = (1.0 - attention_mask.to(hidden.dtype).unsqueeze(1).unsqueeze(2)) \
            * torch.finfo(hidden.dtype).min

        for layer in self.encoder.layer:
            context = layer.attention.self(hidden, additive)
            context = layer.attention.output.dense(context)
            hidden = layer.attention.output.LayerNorm(hidden + context)
            intermediate = layer.intermediate.dense(hidden)
            intermediate = F.gelu(intermediate)
            out = layer.output.dense(intermediate)
            hidden = layer.output.LayerNorm(hidden + out)
        return hidden


def _remap(state: dict) -> dict:
    """Map HuggingFace ``bert.*`` keys onto this module's parameter names.

    Handles both the modern ``LayerNorm.weight/bias`` naming and the legacy
    ``LayerNorm.gamma/beta`` naming still used by some published checkpoints.
    """
    out = {}
    for key, value in state.items():
        if not key.startswith("bert."):
            continue
        new = key[len("bert."):]
        if new.startswith("pooler."):
            continue
        if new.endswith(".gamma"):
            new = new[:-len(".gamma")] + ".weight"
        elif new.endswith(".beta"):
            new = new[:-len(".beta")] + ".bias"
        out[new] = value
    return out


def load_bert(directory: str | Path, device: torch.device | str = "cpu") -> BertEncoder:
    directory = Path(directory)
    with (directory / "config.json").open(encoding="utf-8") as handle:
        cfg = BertConfig(json.load(handle))
    weights = None
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = directory / name
        if not candidate.exists():
            continue
        if name.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError:
                continue
            weights = load_file(str(candidate))
        else:
            weights = torch.load(candidate, map_location="cpu", weights_only=True)
        break
    if weights is None:
        raise FileNotFoundError(f"No BERT weights found in {directory}")
    model = BertEncoder(cfg)
    missing, unexpected = model.load_state_dict(_remap(weights), strict=False)
    if missing:
        raise RuntimeError(f"Missing BERT parameters: {missing[:5]}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


@torch.no_grad()
def encode(model: BertEncoder, text_bert: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    """Encode ``[N, 3, 50]`` token inputs to ``[N, 50, 768]`` features."""
    outputs = []
    device = next(model.parameters()).device
    for start in range(0, len(text_bert), batch_size):
        chunk = text_bert[start:start + batch_size].to(device)
        outputs.append(model(chunk).float().cpu())
    return torch.cat(outputs, dim=0)
