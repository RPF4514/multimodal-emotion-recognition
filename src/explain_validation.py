"""Validation check of whether high gate weights predict model sensitivity.

Gate scores choose the candidate window before occlusion; the comparison is a
same-sample, same-modality random window. This does not validate human meaning.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import torch

from .compare import load_checkpoint
from .data import Normalizer, load_pickle, EmotionDataset, to_device
from .explain import remove_mode
from .paths import ALIGNED, LOGS, OUT, ensure_dirs, selected_checkpoint_path
from .text_bert_encoder import load_bert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=str,
                        default=str(LOGS / "model_selection_bertmask_20260924.json"))
    parser.add_argument("--bert-dir", type=Path, default=OUT / "bert" / "bert-base-uncased")
    parser.add_argument("--output-tag", default="bertmask_20260924")
    args = parser.parse_args()
    ensure_dirs()
    torch.set_num_threads(4)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(selected_checkpoint_path(selection), device)
    bert_encoder = load_bert(args.bert_dir, device)
    normalizer = Normalizer.load(OUT / "normalizer.npz")
    source = load_pickle(ALIGNED)
    dataset = EmotionDataset(source["valid"], normalizer, labeled=True)
    rng = np.random.default_rng(42)
    sample_indices = np.linspace(0, len(dataset) - 1, 120, dtype=int)
    rows = []
    with torch.no_grad():
        for index in sample_indices:
            batch = to_device({k: v.unsqueeze(0) for k, v in dataset[index].items()}, device)
            full = model(batch)
            if full["weights"] is None:
                raise RuntimeError("Validation requires a gated model")
            cls = int(full["logits"].argmax(-1)[0])
            reference = float(torch.softmax(full["logits"], -1)[0, cls])
            length = int(batch["valid"].sum())
            weights = full["weights"][0, :length].cpu().numpy()
            for mode in range(3):
                candidates = [(start, min(length, start + 4)) for start in range(length)
                              if bool(batch["content"][0, start]) and bool(batch["observed"][0, start, mode])]
                if len(candidates) < 2:
                    continue
                score = [float(weights[start:end, mode].mean()) for start, end in candidates]
                top_index = int(np.argmax(score))
                alternatives = [i for i in range(len(candidates)) if i != top_index]
                random_index = int(rng.choice(alternatives))
                drops = []
                for chosen in (top_index, random_index):
                    start, end = candidates[chosen]
                    damaged = model(remove_mode(batch, mode, start, end, bert_encoder))
                    drops.append(reference - float(torch.softmax(damaged["logits"], -1)[0, cls]))
                rows.append({"index": int(index), "mode": mode,
                             "gate_selected_drop": drops[0], "random_drop": drops[1],
                             "difference": drops[0] - drops[1]})
    differences = np.asarray([r["difference"] for r in rows])
    by_sample = np.asarray([np.mean([r["difference"] for r in rows if r["index"] == index])
                            for index in sample_indices if any(r["index"] == index for r in rows)])
    bootstrap = np.asarray([rng.choice(by_sample, size=len(by_sample), replace=True).mean()
                            for _ in range(2000)])
    report = {"split": "valid", "sample_count": len(sample_indices), "paired_comparisons": len(rows),
              "selected_checkpoint": str(selected_checkpoint_path(selection)),
              "occlusion_protocol": "pre-BERT token masking and full contextual re-encoding",
              "window_steps": 4, "mean_gate_selected_drop": float(np.mean([r["gate_selected_drop"] for r in rows])),
              "mean_random_drop": float(np.mean([r["random_drop"] for r in rows])),
              "mean_paired_difference": float(differences.mean()),
              "sample_bootstrap_95pct_interval": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
              "gate_better_fraction": float(np.mean(differences > 0)),
              "interpretation": "Gate-selected windows are chosen without observing deletion scores. "
                                "This checks gate faithfulness, not whether explanations match human reasoning.",
              "rows": rows}
    (LOGS / f"explanation_validation_{args.output_tag}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
