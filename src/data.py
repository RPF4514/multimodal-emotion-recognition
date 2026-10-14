from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .paths import ALIGNED, OUT


def load_pickle(path: Path):
    # Pickle is executable; only open the supplied competition files.
    with path.open("rb") as handle:
        return pickle.load(handle)


def observed_mask(features: np.ndarray) -> np.ndarray:
    return np.isfinite(features).all(axis=-1) & (np.max(np.abs(features), axis=-1) > 1e-8)


def infer_masks(record: dict, is_training_split: bool = True):
    bert = np.asarray(record["text_bert"])
    if bert.ndim == 2:
        bert = bert[None]
    text_obs = bert[:, 1, :].astype(bool)
    audio = np.asarray(record["audio"])
    vision = np.asarray(record["vision"])
    if audio.ndim == 2:
        audio = audio[None]
    if vision.ndim == 2:
        vision = vision[None]
    audio_obs = observed_mask(audio)
    vision_obs = observed_mask(vision)
    # In all supplied aligned_50 splits, including attachment 3, the BERT
    # attention mask is contiguous and ends at [SEP]. Audio/vision zeros are
    # observations, not evidence that the structural sequence has ended.
    structural = text_obs.copy()
    observed = np.stack([text_obs, audio_obs, vision_obs], axis=-1) & structural[..., None]
    return structural.astype(bool), observed.astype(bool)


@dataclass
class Normalizer:
    audio_mean: np.ndarray
    audio_std: np.ndarray
    vision_mean: np.ndarray
    vision_std: np.ndarray

    @classmethod
    def fit(cls, train: dict):
        valid, obs = infer_masks(train, True)
        arrays = []
        for key, mode in (("audio", 1), ("vision", 2)):
            x = np.asarray(train[key], dtype=np.float32)
            selected = x[obs[..., mode] & valid]
            mean = selected.mean(axis=0)
            std = selected.std(axis=0)
            std = np.where(std < 1e-5, 1.0, std)
            arrays.extend([mean, std])
        return cls(*arrays)

    def save(self, path: Path):
        np.savez_compressed(path, audio_mean=self.audio_mean, audio_std=self.audio_std,
                            vision_mean=self.vision_mean, vision_std=self.vision_std)

    @classmethod
    def load(cls, path: Path):
        data = np.load(path)
        return cls(*(data[key] for key in ("audio_mean", "audio_std", "vision_mean", "vision_std")))

    def apply(self, audio: np.ndarray, vision: np.ndarray, observed: np.ndarray):
        audio = (audio.astype(np.float32) - self.audio_mean) / self.audio_std
        vision = (vision.astype(np.float32) - self.vision_mean) / self.vision_std
        audio = np.where(observed[..., 1, None], audio, 0.0)
        vision = np.where(observed[..., 2, None], vision, 0.0)
        return audio.astype(np.float32), vision.astype(np.float32)


class EmotionDataset(Dataset):
    def __init__(self, record: dict, normalizer: Normalizer, labeled: bool = True):
        bert = np.asarray(record["text_bert"])
        if bert.ndim == 2:
            bert = bert[None]
        raw_ids = record.get("id", range(len(bert)))
        self.ids = [raw_ids] if isinstance(raw_ids, str) else list(raw_ids)
        valid, observed = infer_masks(record, labeled)
        audio = np.asarray(record["audio"])
        vision = np.asarray(record["vision"])
        if audio.ndim == 2:
            audio = audio[None]
        if vision.ndim == 2:
            vision = vision[None]
        audio, vision = normalizer.apply(audio, vision, observed)
        self.token_ids = torch.from_numpy(bert[:, 0, :].copy()).long()
        self.text_mask = torch.from_numpy(bert[:, 1, :].copy()).bool()
        self.token_type_ids = torch.from_numpy(bert[:, 2, :].copy()).long()
        content = valid & (bert[:, 0, :] != 101) & (bert[:, 0, :] != 102)
        self.content = torch.from_numpy(content.copy())
        self.audio = torch.from_numpy(audio.copy())
        self.vision = torch.from_numpy(vision.copy())
        self.valid = torch.from_numpy(valid.copy())
        self.observed = torch.from_numpy(observed.copy())
        if "text" in record:
            teacher = np.asarray(record["text"], dtype=np.float32)
            if teacher.ndim == 2:
                teacher = teacher[None]
            if teacher.shape != (len(bert), bert.shape[-1], 768):
                raise ValueError(f"Unexpected text feature shape {teacher.shape}; "
                                 f"expected {(len(bert), bert.shape[-1], 768)}")
            self.teacher_text = torch.from_numpy(teacher.copy())
        else:
            self.teacher_text = None
        self.labeled = labeled
        if labeled:
            self.cls = torch.from_numpy(np.asarray(record["classification_labels"], dtype=np.int64))
            self.reg = torch.from_numpy(np.asarray(record["regression_labels"], dtype=np.float32))

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, index):
        out = {key: getattr(self, key)[index] for key in
               ("token_ids", "text_mask", "token_type_ids", "content", "audio", "vision", "valid", "observed")}
        if self.teacher_text is not None:
            out["teacher_text"] = self.teacher_text[index]
        if self.labeled:
            out["cls"] = self.cls[index]
            out["reg"] = self.reg[index]
        return out


def load_datasets():
    raw = load_pickle(ALIGNED)
    normalizer = Normalizer.fit(raw["train"])
    normalizer.save(OUT / "normalizer.npz")
    datasets = {split: EmotionDataset(raw[split], normalizer, True) for split in ("train", "valid", "test")}
    return datasets, normalizer


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def mask_batch(batch: dict, rng: np.random.Generator, patterns=None,
               lengths=(0.1, 0.3, 0.5), position="random"):
    if patterns is None:
        patterns = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))
    result = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    bsize, tmax = result["valid"].shape
    for i in range(bsize):
        valid_positions = torch.where(result["content"][i])[0]
        if not len(valid_positions):
            continue
        n = len(valid_positions)
        pattern = patterns[int(rng.integers(len(patterns)))]
        frac = lengths[int(rng.integers(len(lengths)))]
        span = max(1, round(n * frac))
        if position == "start":
            start = 0
        elif position == "middle":
            start = max(0, (n - span) // 2)
        elif position == "end":
            start = n - span
        else:
            start = int(rng.integers(max(1, n - span + 1)))
        indices = valid_positions[start:start + span]
        for mode in pattern:
            result["observed"][i, indices, mode] = False
            if mode == 0:
                result["text_mask"][i, indices] = False
                result["token_ids"][i, indices] = 0
                if "teacher_text" in result:
                    result["teacher_text"][i, indices] = 0
            elif mode == 1:
                result["audio"][i, indices] = 0
            else:
                result["vision"][i, indices] = 0
    return result


def deterministic_mask(batch: dict, pattern: tuple[int, ...], fraction: float,
                       position: str = "mixed"):
    out = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    for i in range(len(out["valid"])):
        indices = torch.where(out["content"][i])[0]
        n = len(indices)
        if not n:
            continue
        span = max(1, round(n * fraction))
        place = (i % 3) if position == "mixed" else {"start": 0, "middle": 1, "end": 2}[position]
        start = [0, max(0, (n - span) // 2), n - span][place]
        for mode in pattern:
            selected = indices[start:start + span]
            out["observed"][i, selected, mode] = False
            if mode == 0:
                out["text_mask"][i, selected] = False
                out["token_ids"][i, selected] = 0
                if "teacher_text" in out:
                    out["teacher_text"][i, selected] = 0
            elif mode == 1:
                out["audio"][i, selected] = 0
            else:
                out["vision"][i, selected] = 0
    return out


def reencode_masked_text(batch: dict, bert_encoder) -> dict:
    """Recompute every contextual text vector after a token-level mask.

    Merely zeroing masked rows of a precomputed BERT tensor leaks the removed
    words through [CLS], [SEP], and the surviving contextual token vectors.
    ``valid`` and ``content`` remain the *original* structural masks.
    """
    if "teacher_text" not in batch:
        return batch
    if bert_encoder is None:
        raise ValueError("A frozen BERT encoder is required for contextual text masking")
    bert_input = torch.stack((batch["token_ids"], batch["text_mask"].long(),
                              batch["token_type_ids"]), dim=1)
    with torch.inference_mode():
        contextual = bert_encoder(bert_input).float()
    # A missing text position has no usable feature, regardless of BERT's
    # nonzero output for its [PAD] embedding.
    batch["teacher_text"] = contextual.masked_fill(~batch["observed"][..., 0, None], 0)
    return batch
