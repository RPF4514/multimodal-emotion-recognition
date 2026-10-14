"""Paired validation comparison under the fixed content-only mask protocol."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .compare import load_checkpoint
from .data import deterministic_mask, load_datasets, reencode_masked_text, to_device
from .metrics import all_metrics
from .paths import CHECKPOINTS, LOGS, ensure_dirs

SEEDS = (42, 43, 44)
PATTERNS = {"T": (0,), "A": (1,), "V": (2,),
            "TA": (0, 1), "TV": (0, 2), "AV": (1, 2)}
FRACTIONS = (0.1, 0.3, 0.5)
SCENES = [("complete", None, None)] + [
    (f"{name}_{int(fraction * 100)}", pattern, fraction)
    for name, pattern in PATTERNS.items() for fraction in FRACTIONS
]
GROUPS = {
    "student_gated": "gated_bigru_content24_seed{seed}.pt",
    "emotion_gated": "gated_bigru_emotion24_seed{seed}.pt",
    "emotion_async": "async_late_bigru_emotion24_seed{seed}.pt",
}


def predict(model, loader, device, pattern, fraction, bert_encoder=None):
    truth_cls, truth_reg, pred_cls, pred_reg = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if pattern is not None:
                batch = deterministic_mask(batch, pattern, fraction, position="mixed")
            batch = to_device(batch, device)
            if (pattern is not None and 0 in pattern and bert_encoder is not None
                    and "teacher_text" in batch):
                batch = reencode_masked_text(batch, bert_encoder)
            output = model(batch)
            truth_cls.extend(batch["cls"].cpu().numpy().tolist())
            truth_reg.extend(batch["reg"].cpu().numpy().tolist())
            pred_cls.extend(output["logits"].argmax(-1).cpu().numpy().tolist())
            pred_reg.extend(output["strength"].cpu().numpy().tolist())
    return (np.asarray(truth_cls, dtype=np.int8), np.asarray(truth_reg, dtype=np.float32),
            np.asarray(pred_cls, dtype=np.int8), np.asarray(pred_reg, dtype=np.float32))


def macro_f1(truth, predicted):
    cm = np.bincount(3 * truth.astype(np.int16) + predicted.astype(np.int16),
                     minlength=9).reshape(3, 3)
    precision = np.diag(cm) / np.maximum(cm.sum(0), 1)
    recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(f1.mean())


def aggregate_f1(truth, predictions, indices):
    # predictions: [seed, scene, sample]
    values = []
    sampled_truth = truth[indices]
    for seed_matrix in predictions:
        complete = macro_f1(sampled_truth, seed_matrix[0, indices])
        robust = np.mean([macro_f1(sampled_truth, row[indices])
                          for row in seed_matrix[1:]])
        values.append((complete + robust) / 2)
    return float(np.mean(values))


def cluster_bootstrap(truth, base, candidate, video_ids, repeats=300):
    clusters = defaultdict(list)
    for index, video_id in enumerate(video_ids):
        clusters[video_id].append(index)
    groups = [np.asarray(indices, dtype=np.int32) for indices in clusters.values()]
    rng = np.random.default_rng(20260924)
    differences = []
    for _ in range(repeats):
        drawn = rng.integers(0, len(groups), len(groups))
        indices = np.concatenate([groups[index] for index in drawn])
        differences.append(aggregate_f1(truth, candidate, indices) -
                           aggregate_f1(truth, base, indices))
    return {"unit": "validation video_id", "clusters": len(groups),
            "repeats": repeats, "difference_ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
            "difference_median": float(np.median(differences))}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ensure_dirs()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, _ = load_datasets()
    valid = datasets["valid"]
    loader = DataLoader(valid, batch_size=64, shuffle=False)
    truth_cls = valid.cls.numpy().astype(np.int8)
    truth_reg = valid.reg.numpy().astype(np.float32)
    video_ids = [str(value).split("$_$", 1)[0] for value in valid.ids]
    all_predictions = {}
    summary = {}
    detail = {}
    for group, filename in GROUPS.items():
        cls_seeds, reg_seeds, rows = [], [], []
        detail[group] = {}
        for seed in SEEDS:
            path = CHECKPOINTS / filename.format(seed=seed)
            model, saved = load_checkpoint(path, device)
            if saved.get("mask_protocol") != "content_only_20260924":
                raise ValueError(f"Unexpected mask protocol in {path}")
            cls_scenes, reg_scenes, metrics = [], [], {}
            for name, pattern, fraction in SCENES:
                actual_cls, actual_reg, pred_cls, pred_reg = predict(
                    model, loader, device, pattern, fraction)
                if not np.array_equal(actual_cls, truth_cls) or not np.allclose(actual_reg, truth_reg):
                    raise ValueError("Evaluation order or truth labels changed between scenarios.")
                cls_scenes.append(pred_cls)
                reg_scenes.append(pred_reg)
                metrics[name] = all_metrics(truth_cls, pred_cls, truth_reg, pred_reg)
            cls_seeds.append(np.stack(cls_scenes))
            reg_seeds.append(np.stack(reg_scenes))
            robust_f1 = float(np.mean([metrics[name]["macro_f1"] for name, _, _ in SCENES[1:]]))
            robust_mae = float(np.mean([metrics[name]["mae"] for name, _, _ in SCENES[1:]]))
            row = {"group": group, "seed": seed, "best_epoch": int(saved["epoch"]),
                   "clean_macro_f1": metrics["complete"]["macro_f1"],
                   "robust_macro_f1": robust_f1,
                   "clean_mae": metrics["complete"]["mae"],
                   "robust_mae": robust_mae,
                   "clean_pearson": metrics["complete"]["pearson"],
                   "robust_pearson": float(np.mean([metrics[name]["pearson"]
                                                    for name, _, _ in SCENES[1:]])),
                   "aggregate_f1": (metrics["complete"]["macro_f1"] + robust_f1) / 2,
                   "checkpoint_bytes": path.stat().st_size,
                   "checkpoint_sha256": file_sha256(path)}
            rows.append(row)
            detail[group][str(seed)] = metrics
            print(json.dumps(row), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        all_predictions[group] = {"class": np.stack(cls_seeds), "strength": np.stack(reg_seeds)}
        summary[group] = {"rows": rows, "mean": {
            key: float(np.mean([row[key] for row in rows])) for key in
            ("clean_macro_f1", "robust_macro_f1", "clean_mae", "robust_mae",
             "clean_pearson", "robust_pearson", "aggregate_f1")},
            "std": {key: float(np.std([row[key] for row in rows], ddof=1)) for key in
                    ("clean_macro_f1", "robust_macro_f1", "clean_mae", "robust_mae", "aggregate_f1")}}
    packed = {"truth_cls": truth_cls, "truth_reg": truth_reg,
              "scenes": np.asarray([item[0] for item in SCENES]),
              "video_ids": np.asarray(video_ids)}
    for group, tensors in all_predictions.items():
        packed[f"{group}_class"] = tensors["class"]
        packed[f"{group}_strength"] = tensors["strength"]
    np.savez_compressed(LOGS / "candidate_predictions_20260924.npz", **packed)
    checks = {}
    baseline = summary["student_gated"]["mean"]
    for group in ("emotion_gated", "emotion_async"):
        candidate = summary[group]["mean"]
        paired = cluster_bootstrap(truth_cls, all_predictions["student_gated"]["class"],
                                   all_predictions[group]["class"], video_ids)
        difference = candidate["aggregate_f1"] - baseline["aggregate_f1"]
        clean_mae_delta = candidate["clean_mae"] - baseline["clean_mae"]
        robust_mae_delta = candidate["robust_mae"] - baseline["robust_mae"]
        size_ok = max(row["checkpoint_bytes"] for row in summary[group]["rows"]) <= 50 * 1024 * 1024
        passed = bool(difference >= 0.01 and paired["difference_ci95"][0] > 0 and
                      clean_mae_delta <= 0.01 and robust_mae_delta <= 0.01 and size_ok)
        checks[group] = {"aggregate_f1_difference": difference,
                         "clean_mae_difference": clean_mae_delta,
                         "robust_mae_difference": robust_mae_delta,
                         "paired_bootstrap": paired, "checkpoint_under_50MiB": size_ok,
                         "predeclared_rule_passed": passed}
    passed_groups = [group for group in checks if checks[group]["predeclared_rule_passed"]]
    selected = (max(passed_groups, key=lambda name: summary[name]["mean"]["aggregate_f1"])
                if passed_groups else "student_gated")
    chosen = max(summary[selected]["rows"], key=lambda row: row["aggregate_f1"])
    decision = {"selected_group": selected, "selected_model": (
                    "async_late_bigru" if selected == "emotion_async" else "gated_bigru"),
                "selected_seed": chosen["seed"],
                "selected_checkpoint": str(CHECKPOINTS /
                    GROUPS[selected].format(seed=chosen["seed"])),
                "selection_basis": "Predeclared validation aggregate F1, cluster bootstrap, MAE and size",
                "mask_protocol": "content_only_20260924", "checks": checks,
                "summary": summary, "scenes": [item[0] for item in SCENES],
                "test_evaluated": False}
    (LOGS / "model_selection_20260924.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS / "candidate_validation_20260924.json").write_text(
        json.dumps({"summary": summary, "scenario_metrics": detail,
                    "checks": checks, "mask_protocol": "content_only_20260924",
                    "split": "valid"}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (LOGS / "candidate_comparison_20260924.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["student_gated"]["rows"][0]))
        writer.writeheader()
        for group in GROUPS:
            writer.writerows(summary[group]["rows"])
    print(json.dumps({"decision": str(LOGS / "model_selection_20260924.json"),
                      "selected": selected, "checks": checks}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
