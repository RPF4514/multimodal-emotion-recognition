"""Validation-only comparison after pre-BERT text masking; never opens test."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .compare import load_checkpoint
from .data import load_datasets
from .evaluate_candidates import SCENES, cluster_bootstrap, predict
from .metrics import all_metrics
from .paths import CHECKPOINTS, LOGS, OUT, ensure_dirs
from .text_bert_encoder import load_bert

SEEDS = (42, 43, 44)
PROTOCOL = "bert_reencoded_20260924"
GROUPS = {
    "legacy_bridge_corrected_eval": "gated_bigru_bridge24_seed{seed}.pt",
    "retrained_bridge": "gated_bigru_bertmask24_seed{seed}.pt",
    "retrained_teacher": "gated_bigru_teacher_bertmask24_seed{seed}.pt",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--bert-dir", type=Path, default=OUT / "bert" / "bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    ensure_dirs()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert = load_bert(args.bert_dir, device)
    datasets, _ = load_datasets()
    valid = datasets["valid"]
    loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False)
    truth_cls = valid.cls.numpy().astype(np.int8)
    truth_reg = valid.reg.numpy().astype(np.float32)
    video_ids = [str(item).split("$_$", 1)[0] for item in valid.ids]
    summary, details, predictions = {}, {}, {}
    for group in args.groups:
        rows, seed_cls = [], []
        details[group] = {}
        for seed in SEEDS:
            path = CHECKPOINTS / GROUPS[group].format(seed=seed)
            model, saved = load_checkpoint(path, device)
            expected = "content_only_20260924" if group.startswith("legacy") else PROTOCOL
            if saved.get("mask_protocol") != expected:
                raise ValueError(f"Unexpected mask protocol in {path}: {saved.get('mask_protocol')}")
            scene_metrics, scene_predictions = {}, []
            for name, pattern, fraction in SCENES:
                actual_cls, actual_reg, predicted_cls, predicted_reg = predict(
                    model, loader, device, pattern, fraction, bert)
                if not np.array_equal(actual_cls, truth_cls) or not np.allclose(actual_reg, truth_reg):
                    raise ValueError("Validation labels/order changed")
                scene_metrics[name] = all_metrics(truth_cls, predicted_cls,
                                                  truth_reg, predicted_reg)
                scene_predictions.append(predicted_cls)
            robust_names = [name for name, _, _ in SCENES[1:]]
            robust_f1 = float(np.mean([scene_metrics[name]["macro_f1"] for name in robust_names]))
            robust_mae = float(np.mean([scene_metrics[name]["mae"] for name in robust_names]))
            complete = scene_metrics["complete"]
            row = {"group": group, "seed": seed, "epoch": int(saved["epoch"]),
                   "complete_accuracy": complete["accuracy"],
                   "complete_macro_f1": complete["macro_f1"],
                   "robust_macro_f1": robust_f1,
                   "complete_mae": complete["mae"], "robust_mae": robust_mae,
                   "complete_pearson": complete["pearson"],
                   "aggregate_f1": (complete["macro_f1"] + robust_f1) / 2,
                   "checkpoint_bytes": path.stat().st_size}
            rows.append(row)
            details[group][str(seed)] = scene_metrics
            seed_cls.append(np.stack(scene_predictions))
            print(json.dumps(row), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        summary[group] = {"rows": rows,
                          "mean": {key: float(np.mean([r[key] for r in rows])) for key in (
                              "complete_accuracy", "complete_macro_f1", "robust_macro_f1",
                              "complete_mae", "robust_mae", "complete_pearson", "aggregate_f1")},
                          "std": {key: float(np.std([r[key] for r in rows], ddof=1)) for key in (
                              "complete_macro_f1", "robust_macro_f1", "complete_mae",
                              "robust_mae", "aggregate_f1")}}
        predictions[group] = np.stack(seed_cls)
    comparisons = {}
    pairs = [("legacy_bridge_corrected_eval", group) for group in args.groups
             if group != "legacy_bridge_corrected_eval"]
    pairs.append(("retrained_bridge", "retrained_teacher"))
    for base, group in pairs:
        if base not in predictions or group not in predictions:
            continue
        paired = cluster_bootstrap(truth_cls, predictions[base], predictions[group],
                                   video_ids, repeats=500)
        delta = summary[group]["mean"]["aggregate_f1"] - summary[base]["mean"]["aggregate_f1"]
        comparisons[f"{group}_vs_{base}"] = {
            "against": base, "aggregate_f1_difference": delta,
            "paired_video_bootstrap": paired,
            "complete_mae_difference": (summary[group]["mean"]["complete_mae"]
                                        - summary[base]["mean"]["complete_mae"]),
            "robust_mae_difference": (summary[group]["mean"]["robust_mae"]
                                      - summary[base]["mean"]["robust_mae"])}
    report = {"split": "valid", "test_evaluated": False, "mask_protocol": PROTOCOL,
              "text_mask_method": "zero token IDs and attention before frozen BERT; recompute all positions",
              "groups": summary, "scenario_metrics": details, "comparisons": comparisons,
              "note": "Historical test results and old content-only missing-scene scores are not reused for selection."}
    output = LOGS / "bertmask_validation_20260924.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (LOGS / "bertmask_validation_20260924.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(summary.values()))["rows"][0]))
        writer.writeheader()
        for group in args.groups:
            writer.writerows(summary[group]["rows"])
    if "legacy_bridge_corrected_eval" in summary and len(args.groups) > 1:
        base = "legacy_bridge_corrected_eval"
        eligibility = {}
        for group in args.groups:
            if group == base:
                continue
            comparison = comparisons[f"{group}_vs_{base}"]
            eligibility[group] = bool(
                comparison["aggregate_f1_difference"] >= 0.01
                and comparison["paired_video_bootstrap"]["difference_ci95"][0] > 0
                and comparison["complete_mae_difference"] <= 0.01
                and comparison["robust_mae_difference"] <= 0.01
                and all(row["checkpoint_bytes"] <= 50 * 1024 * 1024
                        for row in summary[group]["rows"]))
        qualified = [group for group, passed in eligibility.items() if passed]
        selected_group = (max(qualified, key=lambda group: summary[group]["mean"]["aggregate_f1"])
                          if qualified else base)
        selected_row = max(summary[selected_group]["rows"], key=lambda row: row["aggregate_f1"])
        checkpoint = CHECKPOINTS / GROUPS[selected_group].format(seed=selected_row["seed"])
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        decision = {"selected_group": selected_group, "selected_checkpoint": str(checkpoint),
                    "selected_seed": selected_row["seed"], "text_source": saved["text_source"],
                    "mask_protocol": saved["mask_protocol"], "eligibility": eligibility,
                    "selection_rule": "mean aggregate F1 >= baseline +0.01, positive video-cluster paired CI, "
                                      "both MAE deltas <=0.01, each checkpoint <=50MiB; "
                                      "otherwise retain baseline",
                    "validation_report": str(output), "test_evaluated": False}
        decision_path = LOGS / "model_selection_bertmask_20260924.json"
        if decision_path.exists():
            raise FileExistsError(decision_path)
        decision_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"selection": str(decision_path), "selected_group": selected_group,
                          "eligibility": eligibility}, ensure_ascii=False), flush=True)
    print(json.dumps({"report": str(output), "summary": {k: v["mean"] for k, v in summary.items()},
                      "comparisons": comparisons}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
