from __future__ import annotations

import torch
from torch import nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(x.dtype).unsqueeze(-1)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


class TextStudent(nn.Module):
    """A small token encoder distilled only from supplied training-set text features."""

    def __init__(self, vocab_size=30522, dim=128, max_length=50, num_layers=1):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.position = nn.Embedding(max_length, dim)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=256,
                                           dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.teacher_head = nn.Linear(dim, 768)

    def features(self, token_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)[None, :]
        x = self.embedding(token_ids) + self.position(positions)
        # For fully missing text, one dummy key prevents attention NaNs.
        safe_mask = text_mask.clone()
        missing = ~safe_mask.any(dim=1)
        if missing.any():
            safe_mask[missing, 0] = True
        x = self.encoder(x, src_key_padding_mask=~safe_mask)
        return x * text_mask.unsqueeze(-1)

    def forward(self, token_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        return self.teacher_head(self.features(token_ids, text_mask))


class EmotionModel(nn.Module):
    """Fusion model over text / audio / vision.

    ``text_source`` selects how the text slot is filled:

    * ``student``  - frozen distilled token encoder (works everywhere,
      but caps the achievable accuracy).
    * ``teacher``  - the supplied precomputed 768-d text features (strong, but
      only attachments 1/2/4 carry them).
    * ``bridge``   - a single 768-d text slot that is filled by the precomputed
      features when present and by a frozen student encoder plus a learned
      768-d adapter when they are absent. The adapter is trained both as a
      reconstruction target of the precomputed features and as a randomly
      swapped substitute inside the fusion trunk, so one checkpoint serves all
      four attachments without degrading the main protocol.
    """

    def __init__(self, kind: str, text_student: TextStudent, hidden=64, dropout=0.2,
                 text_source="student", train_text_student=False, bridge_dropout=0.25,
                 class_mode="softmax"):
        super().__init__()
        self.kind = kind
        self.text_source = text_source
        self.class_mode = class_mode
        if class_mode not in ("softmax", "ordinal"):
            raise ValueError(f"Unknown class mode: {class_mode}")
        self.train_text_student = train_text_student
        self.bridge_dropout = bridge_dropout
        self.text_student = text_student
        for param in self.text_student.parameters():
            param.requires_grad_(train_text_student)
        if not train_text_student:
            self.text_student.eval()
        text_dim = text_student.dim if text_source == "student" else 768
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(text_dim, hidden),
                          nn.LayerNorm(hidden), nn.GELU()),
            nn.Sequential(nn.Linear(74, hidden), nn.LayerNorm(hidden), nn.GELU()),
            nn.Sequential(nn.Linear(35, hidden), nn.LayerNorm(hidden), nn.GELU()),
        ])
        if text_source == "bridge":
            # Recovers the 768-d precomputed representation from deployable
            # student features; the one-slot interface stays the same.
            self.text_adapter = nn.Sequential(
                nn.Linear(text_student.dim, 512), nn.LayerNorm(512), nn.GELU(),
                nn.Linear(512, 768), nn.LayerNorm(768), nn.GELU())
        if kind == "baseline_bigru":
            self.temporal = nn.GRU(hidden * 3, hidden, bidirectional=True, batch_first=True)
            output_dim = hidden * 2
        elif kind == "gated_bigru":
            self.mode_rnns = nn.ModuleList([nn.GRU(hidden, hidden // 2,
                                                   bidirectional=True, batch_first=True)
                                             for _ in range(3)])
            self.gate = nn.Linear(hidden, 1)
            self.temporal = nn.GRU(hidden, hidden, bidirectional=True, batch_first=True)
            output_dim = hidden * 2
        elif kind == "gated_transformer":
            self.gate = nn.Linear(hidden, 1)
            self.temporal_position = nn.Embedding(50, hidden)
            layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=hidden * 2,
                                               dropout=dropout, batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            output_dim = hidden
        else:
            raise ValueError(f"Unknown model kind: {kind}")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_dim, 2 if class_mode == "ordinal" else 3)
        self.regressor = nn.Linear(output_dim, 1)

    def train(self, mode=True):
        super().train(mode)
        if not self.train_text_student:
            self.text_student.eval()
        return self

    def forward(self, batch: dict):
        recon = None
        if self.text_source == "bridge":
            with torch.no_grad():
                student = self.text_student.features(batch["token_ids"], batch["text_mask"])
            recovered = self.text_adapter(student) * batch["text_mask"].unsqueeze(-1)
            teacher = batch.get("teacher_text")
            if teacher is None:
                # Attachments without supplied precomputed features (e.g. attachment 3).
                text = recovered
            elif self.training and self.bridge_dropout > 0:
                swap = torch.rand(teacher.shape[0], 1, 1, device=teacher.device) < self.bridge_dropout
                text = torch.where(swap, recovered, teacher)
                weight = batch["text_mask"].unsqueeze(-1).to(recovered.dtype)
                # Count positions *and* feature dimensions; the raw sum runs over both.
                denom = weight.sum().clamp_min(1.0) * recovered.shape[-1]
                target = teacher.detach()
                scale = ((target ** 2 * weight).sum() / denom).sqrt().clamp_min(1e-6)
                # Scale-free relative MSE: ~0 when the bridge reproduces the
                # precomputed features, ~O(1) when it carries no information.
                recon = (((recovered - target) ** 2 * weight).sum() / denom) / (scale ** 2)
            else:
                text = teacher
        elif self.text_source == "teacher":
            text = batch["teacher_text"]
        elif self.train_text_student:
            text = self.text_student.features(batch["token_ids"], batch["text_mask"])
        else:
            with torch.no_grad():
                text = self.text_student.features(batch["token_ids"], batch["text_mask"])
        inputs = [text, batch["audio"], batch["vision"]]
        states = [project(x) for project, x in zip(self.projections, inputs)]
        observed = batch["observed"] & batch["valid"].unsqueeze(-1)
        states = [state * observed[..., m].unsqueeze(-1) for m, state in enumerate(states)]
        weights = None
        if self.kind == "baseline_bigru":
            x = torch.cat(states, dim=-1)
            x, _ = self.temporal(x)
        else:
            if self.kind == "gated_bigru":
                states = [rnn(state)[0] for rnn, state in zip(self.mode_rnns, states)]
            stack = torch.stack(states, dim=2)
            scores = self.gate(stack).squeeze(-1).masked_fill(~observed, -1e4)
            weights = torch.softmax(scores, dim=-1) * observed
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            fused = (stack * weights.unsqueeze(-1)).sum(dim=2)
            if self.kind == "gated_bigru":
                x, _ = self.temporal(fused)
            else:
                positions = torch.arange(fused.shape[1], device=fused.device)[None, :]
                x = self.temporal(fused + self.temporal_position(positions),
                                  src_key_padding_mask=~batch["valid"])
        pooled = self.dropout(masked_mean(x, batch["valid"]))
        logits = self.classifier(pooled)
        if self.class_mode == "ordinal":
            # Ordered classes are Negative < Neutral < Positive. Parameterize
            # cumulative probabilities so P(y >= Positive) <= P(y >= Neutral).
            cumulative_neutral = torch.sigmoid(logits[..., 0])
            cumulative_positive = cumulative_neutral * torch.sigmoid(logits[..., 1])
            probabilities = torch.stack((1 - cumulative_neutral,
                                         cumulative_neutral - cumulative_positive,
                                         cumulative_positive), dim=-1)
            logits = probabilities.clamp_min(1e-8).log()
        strength = 3 * torch.tanh(self.regressor(pooled).squeeze(-1))
        return {"logits": logits, "strength": strength, "weights": weights, "recon": recon}


class TextOnlyEmotionModel(nn.Module):
    """Independently trained text baseline using the same TextStudent encoder."""

    def __init__(self, text_student: TextStudent, hidden=64, dropout=0.2,
                 text_source="student"):
        super().__init__()
        self.kind = "text_only_bigru"
        self.text_source = text_source
        self.text_student = text_student
        for parameter in self.text_student.parameters():
            parameter.requires_grad_(False)
        self.text_student.eval()
        input_dim = 768 if text_source == "teacher" else text_student.dim
        self.projection = nn.Sequential(nn.Linear(input_dim, hidden),
                                        nn.LayerNorm(hidden), nn.GELU())
        self.temporal = nn.GRU(hidden, hidden, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, 3)
        self.regressor = nn.Linear(hidden * 2, 1)

    def train(self, mode=True):
        super().train(mode)
        self.text_student.eval()
        return self

    def forward(self, batch: dict):
        if self.text_source == "teacher":
            text = batch["teacher_text"]
        else:
            with torch.no_grad():
                text = self.text_student.features(batch["token_ids"], batch["text_mask"])
        observed = batch["observed"][..., 0] & batch["valid"]
        x = self.projection(text) * observed.unsqueeze(-1)
        x, _ = self.temporal(x)
        pooled = self.dropout(masked_mean(x, batch["valid"]))
        return {"logits": self.classifier(pooled),
                "strength": 3 * torch.tanh(self.regressor(pooled).squeeze(-1)),
                "weights": None}


class AsyncLateEmotionModel(nn.Module):
    """Separate modal time encoders with sample-level, quality-aware fusion."""

    def __init__(self, text_student: TextStudent, hidden=64, dropout=0.2):
        super().__init__()
        self.kind = "async_late_bigru"
        self.text_source = "student"
        self.text_student = text_student
        for parameter in self.text_student.parameters():
            parameter.requires_grad_(False)
        self.text_student.eval()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU())
            for dim in (text_student.dim, 74, 35)
        ])
        self.mode_rnns = nn.ModuleList([
            nn.GRU(hidden, hidden // 2, bidirectional=True, batch_first=True)
            for _ in range(3)
        ])
        self.gate = nn.Linear(hidden + 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 3)
        self.regressor = nn.Linear(hidden, 1)

    def train(self, mode=True):
        super().train(mode)
        self.text_student.eval()
        return self

    def forward(self, batch: dict):
        with torch.no_grad():
            text = self.text_student.features(batch["token_ids"], batch["text_mask"])
        inputs = (text, batch["audio"], batch["vision"])
        content = batch["content"]
        observed = batch["observed"] & content.unsqueeze(-1)
        representations = []
        for mode, value in enumerate(inputs):
            state = self.projections[mode](value) * observed[..., mode].unsqueeze(-1)
            state, _ = self.mode_rnns[mode](state)
            representations.append(masked_mean(state, observed[..., mode]))
        stack = torch.stack(representations, dim=1)
        quality = (observed.sum(dim=1).to(stack.dtype) /
                   content.sum(dim=1, keepdim=True).clamp_min(1).to(stack.dtype))
        available = observed.any(dim=1)
        gate_input = torch.cat((stack, quality.unsqueeze(-1)), dim=-1)
        scores = self.gate(gate_input).squeeze(-1).masked_fill(~available, -1e4)
        weights = torch.softmax(scores, dim=-1) * available
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        fused = self.dropout((stack * weights.unsqueeze(-1)).sum(dim=1))
        return {"logits": self.classifier(fused),
                "strength": 3 * torch.tanh(self.regressor(fused).squeeze(-1)),
                "weights": weights.unsqueeze(1).expand(-1, batch["valid"].shape[1], -1)}
