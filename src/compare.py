from __future__ import annotations

import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import deterministic_mask, load_datasets, to_device
from .metrics import all_metrics
from .models import AsyncLateEmotionModel, EmotionModel, TextOnlyEmotionModel, TextStudent
from .paths import CHECKPOINTS, FIGURES, LOGS, ensure_dirs
from .train import evaluate_loader

MODELS = ("baseline_bigru", "gated_bigru", "gated_transformer")
SEEDS = (42, 43, 44)
PATTERNS = {"T": (0,), "A": (1,), "V": (2,),
            "TA": (0, 1), "TV": (0, 2), "AV": (1, 2)}
FRACTIONS = (0.1, 0.3, 0.5)


def load_checkpoint(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    student_config = saved.get("student_config", {})
    student = TextStudent(dim=student_config.get("dim", 128),
                          num_layers=student_config.get("num_layers", 1))
    if saved["model"] == "text_only_bigru":
        model = TextOnlyEmotionModel(student, hidden=saved["hidden"],
                                     text_source=saved.get("text_source", "student"))
    elif saved["model"] == "async_late_bigru":
        model = AsyncLateEmotionModel(student, hidden=saved["hidden"])
    else:
        model = EmotionModel(saved["model"], student, hidden=saved["hidden"],
                             text_source=saved.get("text_source", "student"),
                             train_text_student=saved.get("train_text_student", False),
                             class_mode=saved.get("class_mode", "softmax"))
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval(), saved


def evaluate_pattern(model, loader, device, pattern, fraction):
    truth_cls, pred_cls, truth_reg, pred_reg = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = deterministic_mask(batch, pattern, fraction)
            batch = to_device(batch, device)
            out = model(batch)
            truth_cls.extend(batch["cls"].cpu().tolist())
            truth_reg.extend(batch["reg"].cpu().tolist())
            pred_cls.extend(out["logits"].argmax(-1).cpu().tolist())
            pred_reg.extend(out["strength"].cpu().tolist())
    return all_metrics(truth_cls, pred_cls, truth_reg, pred_reg)


def evaluate_grid(model, loader, device):
    return {f"{name}_{int(fraction * 100)}": evaluate_pattern(model, loader, device, pattern, fraction)
            for name, pattern in PATTERNS.items() for fraction in FRACTIONS}


def robust_summary(grid):
    return {metric: float(np.mean([row[metric] for row in grid.values()]))
            for metric in ("accuracy", "macro_f1", "weighted_f1", "mae", "rmse", "pearson")}


def plot_comparison(rows):
    models = list(MODELS)
    values = [np.mean([x["valid_robust_macro_f1"] for x in rows if x["model"] == model])
              for model in models]
    clean = [np.mean([x["valid_clean_macro_f1"] for x in rows if x["model"] == model])
             for model in models]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.18, clean, width=0.36, label="Complete")
    ax.bar(x + 0.18, values, width=0.36, label="Missing")
    ax.set_xticks(x, ["BiGRU", "Gated BiGRU", "Gated Transformer"])
    ax.set_ylim(0, max(max(values), max(clean)) + 0.12)
    ax.set_ylabel("Validation macro-F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "validation_model_comparison.png", dpi=160)
    plt.close(fig)


def main():
    ensure_dirs()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, _ = load_datasets()
    valid_loader = DataLoader(datasets["valid"], batch_size=64, shuffle=False)
    rows = []
    details = {}
    for kind in MODELS:
        for seed in SEEDS:
            path = CHECKPOINTS / f"{kind}_seed{seed}.pt"
            if not path.exists():
                raise FileNotFoundError(path)
            model, saved = load_checkpoint(path, device)
            clean = evaluate_loader(model, valid_loader, device)
            grid = evaluate_grid(model, valid_loader, device)
            robust = robust_summary(grid)
            details[f"{kind}_seed{seed}"] = {"clean": clean, "grid": grid, "robust": robust}
            row = {"model": kind, "seed": seed, "best_epoch": saved["epoch"],
                   "valid_clean_macro_f1": clean["macro_f1"],
                   "valid_robust_macro_f1": robust["macro_f1"],
                   "valid_clean_mae": clean["mae"], "valid_robust_mae": robust["mae"],
                   "valid_clean_pearson": clean["pearson"],
                   "valid_robust_pearson": robust["pearson"],
                   "checkpoint_bytes": path.stat().st_size,
                   "selection_score": (clean["macro_f1"] + robust["macro_f1"]) / 2}
            rows.append(row)
            print(json.dumps(row), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    (LOGS / "validation_details.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    with (LOGS / "model_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_comparison(rows)
    summary = {}
    for kind in MODELS:
        group = [row for row in rows if row["model"] == kind]
        summary[kind] = {metric: {"mean": float(np.mean([r[metric] for r in group])),
                                  "std": float(np.std([r[metric] for r in group], ddof=1))}
                         for metric in ("valid_clean_macro_f1", "valid_robust_macro_f1",
                                        "valid_clean_mae", "valid_robust_mae", "selection_score")}
    # Primary choice uses only validation Macro-F1; MAE breaks near ties.
    winner = max(MODELS, key=lambda kind: (summary[kind]["selection_score"]["mean"],
                                           -summary[kind]["valid_robust_mae"]["mean"]))
    best_row = max((r for r in rows if r["model"] == winner), key=lambda r: r["selection_score"])
    decision = {"selected_model": winner, "selected_seed": best_row["seed"],
                "selected_checkpoint": str(CHECKPOINTS / f"{winner}_seed{best_row['seed']}.pt"),
                "selection_basis": "mean validation complete/missing macro-F1 over three seeds",
                "summary": summary, "rows": rows}
    (LOGS / "model_selection.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected_model": winner, "selected_seed": best_row["seed"],
                      "summary": summary}, ensure_ascii=False), flush=True)
    test_loader = DataLoader(datasets["test"], batch_size=64, shuffle=False)
    model, _ = load_checkpoint(CHECKPOINTS / f"{winner}_seed{best_row['seed']}.pt", device)
    test_clean = evaluate_loader(model, test_loader, device)
    test_grid = evaluate_grid(model, test_loader, device)
    test_result = {"model": winner, "seed": best_row["seed"], "clean": test_clean,
                   "robust_average": robust_summary(test_grid), "robust_grid": test_grid}
    (LOGS / "final_test_evaluation.json").write_text(
        json.dumps(test_result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"final_test_clean": test_clean, "final_test_robust_average":
                      test_result["robust_average"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
