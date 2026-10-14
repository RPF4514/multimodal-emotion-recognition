"""One-time post-selection test audit for the locked corrected-mask checkpoint.

This is not a model-selection script. The test split must not drive subsequent
changes to architecture, training, calibration, or selection.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from .compare import load_checkpoint
from .data import EmotionDataset, Normalizer, load_pickle
from .evaluate_candidates import SCENES, predict
from .metrics import all_metrics
from .paths import ALIGNED, LOGS, OUT, ensure_dirs, selected_checkpoint_path
from .text_bert_encoder import load_bert

DECISION = LOGS / "model_selection_bertmask_20260924.json"
OUTPUT = LOGS / "final_test_bertmask_seed43_20260925.json"
# Exact bytes of the packaged checkpoint after path-only metadata anonymization.
# Original-file provenance and unchanged tensors: SOURCE_WEIGHT_AUDIT.json.
LOCKED_CHECKPOINT_SHA256 = "e221cc9ed889bc5af1e311ce19a793b9c5f0dce0b2a1132416363f6b8dbeb167"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_locked_checkpoint(path):
    checkpoint_hash = sha256(path)
    if checkpoint_hash != LOCKED_CHECKPOINT_SHA256:
        raise ValueError("Locked checkpoint hash changed")
    return checkpoint_hash


def main():
    ensure_dirs()
    if OUTPUT.exists():
        raise FileExistsError(f"Final test output already exists: {OUTPUT}")
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    if (decision.get("selected_group") != "retrained_teacher"
            or decision.get("selected_seed") != 43
            or decision.get("mask_protocol") != "bert_reencoded_20260924"
            or decision.get("test_evaluated") is not False):
        raise ValueError("Locked decision does not match the predeclared test audit")
    checkpoint = selected_checkpoint_path(decision)
    checkpoint_hash = verify_locked_checkpoint(checkpoint)
    normalizer_path = OUT / "normalizer.npz"
    if not normalizer_path.is_file():
        raise FileNotFoundError(normalizer_path)
    bert_dir = OUT / "bert" / "bert-base-uncased"
    bert_weights = bert_dir / "pytorch_model.bin"
    if not bert_weights.is_file():
        raise FileNotFoundError(bert_weights)

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, saved = load_checkpoint(checkpoint, device)
    if saved.get("mask_protocol") != decision["mask_protocol"] or saved.get("text_source") != "teacher":
        raise ValueError("Checkpoint metadata differs from locked decision")
    bert = load_bert(bert_dir, device)
    normalizer = Normalizer.load(normalizer_path)
    raw = load_pickle(ALIGNED)
    test = EmotionDataset(raw["test"], normalizer, True)
    loader = DataLoader(test, batch_size=64, shuffle=False)
    truth_cls = test.cls.numpy().astype(np.int8)
    truth_reg = test.reg.numpy().astype(np.float32)
    metrics = {}
    for name, pattern, fraction in SCENES:
        actual_cls, actual_reg, predicted_cls, predicted_reg = predict(
            model, loader, device, pattern, fraction, bert)
        if not np.array_equal(actual_cls, truth_cls) or not np.allclose(actual_reg, truth_reg):
            raise ValueError("Test sample order or labels changed across scenarios")
        metrics[name] = all_metrics(truth_cls, predicted_cls, truth_reg, predicted_reg)
        print(json.dumps({"scene": name, "macro_f1": metrics[name]["macro_f1"],
                          "mae": metrics[name]["mae"]}), flush=True)
    robust_names = [name for name, _, _ in SCENES[1:]]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split": "test", "sample_count": len(test),
        "scope_note": "Post-selection evaluation of the locked seed-43 model; older checkpoints had previously been examined on this same competition test split, so this is not a fully blind dataset-wide estimate. No model changes may follow from this result.",
        "selection_record": str(DECISION), "selection_sha256": sha256(DECISION),
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "normalizer": str(normalizer_path), "normalizer_sha256": sha256(normalizer_path),
        "bert_weights": str(bert_weights), "bert_weights_sha256": sha256(bert_weights),
        "mask_protocol": decision["mask_protocol"],
        "missing_position_protocol": "mixed: start/middle/end by batch sample index modulo 3",
        "text_mask_method": "zero token IDs and attention before frozen BERT; recompute all positions",
        "scenario_metrics": metrics,
        "complete": metrics["complete"],
        "robust_18_equal_mean": {key: float(np.mean([metrics[name][key] for name in robust_names]))
                                 for key in ("accuracy", "macro_f1", "weighted_f1", "mae", "rmse", "pearson")},
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "complete_macro_f1": report["complete"]["macro_f1"],
                      "complete_mae": report["complete"]["mae"],
                      "robust_macro_f1": report["robust_18_equal_mean"]["macro_f1"]}), flush=True)


if __name__ == "__main__":
    main()
