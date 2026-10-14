"""Paired validation-only check of start/middle/end block placement at 30%."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .compare import load_checkpoint
from .data import EmotionDataset, Normalizer, deterministic_mask, load_pickle, reencode_masked_text, to_device
from .evaluate_candidates import PATTERNS, macro_f1
from .metrics import all_metrics
from .paths import ALIGNED, LOGS, OUT, selected_checkpoint_path
from .text_bert_encoder import load_bert

OUTPUT = LOGS / "position_paired_valid_seed43_20260925.json"


def infer(model, loader, device, bert, pattern, position):
    truth_cls, truth_reg, pred_cls, pred_reg = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = deterministic_mask(batch, pattern, 0.3, position=position)
            batch = to_device(batch, device)
            if 0 in pattern:
                batch = reencode_masked_text(batch, bert)
            output = model(batch)
            truth_cls.extend(batch["cls"].cpu().tolist())
            truth_reg.extend(batch["reg"].cpu().tolist())
            pred_cls.extend(output["logits"].argmax(-1).cpu().tolist())
            pred_reg.extend(output["strength"].cpu().tolist())
    return (np.asarray(truth_cls), np.asarray(truth_reg),
            np.asarray(pred_cls), np.asarray(pred_reg))


def paired_video_interval(truth, first, second, video_ids, repeats=500):
    groups = defaultdict(list)
    for i, name in enumerate(video_ids):
        groups[name].append(i)
    clusters = [np.asarray(x, dtype=int) for x in groups.values()]
    rng = np.random.default_rng(20260925)
    differences = []
    for _ in range(repeats):
        indices = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        differences.append(macro_f1(truth[indices], second[indices]) -
                           macro_f1(truth[indices], first[indices]))
    return {"end_minus_start_ci95": np.quantile(differences, [.025, .975]).tolist(),
            "end_minus_start_median": float(np.median(differences)),
            "video_clusters": len(clusters), "bootstrap_repeats": repeats}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    decision = json.loads((LOGS / "model_selection_bertmask_20260924.json").read_text(encoding="utf-8"))
    if decision["selected_seed"] != 43 or decision["mask_protocol"] != "bert_reencoded_20260924":
        raise ValueError("Unexpected selected model")
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, saved = load_checkpoint(selected_checkpoint_path(decision), device)
    if saved["mask_protocol"] != decision["mask_protocol"]:
        raise ValueError("Mask protocol mismatch")
    bert = load_bert(OUT / "bert" / "bert-base-uncased", device)
    normalizer = Normalizer.load(OUT / "normalizer.npz")
    valid = EmotionDataset(load_pickle(ALIGNED)["valid"], normalizer, True)
    loader = DataLoader(valid, batch_size=64, shuffle=False)
    expected_cls = valid.cls.numpy()
    expected_reg = valid.reg.numpy()
    video_ids = [str(x).split("$_$", 1)[0] for x in valid.ids]
    result = {"split": "valid", "sample_count": len(valid), "model_seed": 43,
              "length_fraction": 0.3, "positions": ["start", "middle", "end"],
              "scope_note": "All 728 validation samples paired within each modality combination; does not prove effects for other lengths or the 18-scene mixed-position grid.",
              "patterns": {}}
    for name, pattern in PATTERNS.items():
        metrics, predictions = {}, {}
        for position in ("start", "middle", "end"):
            truth_cls, truth_reg, predicted_cls, predicted_reg = infer(
                model, loader, device, bert, pattern, position)
            if not np.array_equal(truth_cls, expected_cls) or not np.allclose(truth_reg, expected_reg):
                raise ValueError("Paired validation sample order changed")
            metrics[position] = all_metrics(truth_cls, predicted_cls, truth_reg, predicted_reg)
            predictions[position] = predicted_cls
        result["patterns"][name] = {"metrics": metrics,
                                     "end_vs_start": paired_video_interval(
                                         expected_cls, predictions["start"], predictions["end"], video_ids)}
        print(json.dumps({"pattern": name, "macro_f1": {p: metrics[p]["macro_f1"]
                                                          for p in metrics}}), flush=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT)}), flush=True)


if __name__ == "__main__":
    main()
