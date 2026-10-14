"""Regenerate attachment-3 predictions using BERT-derived text features.

Motivation
----------
``attachments 3 (aligned)`` does not store a precomputed floating-point 768-d
``text`` field. It does store ``text_bert``, a verbatim ``bert-base-uncased``
input. Encoding that input with the frozen checkpoint reconstructs the
supplied precomputed ``text`` to float32 precision (mean |delta| = 5e-7,
cosine 1.000000 across attachments 2 and 4 -- see
``text_bert_features/reproduction_report.json``).

For attachment 3 the text slot is filled with this path:

    text_bert  ->  frozen bert-base-uncased  ->  768-d features

This script does exactly that, with no retraining and no re-use of the locked
model's test result. Attachment 4 is included as a control: it ships both
``text`` and ``text_bert``, so its predictions when fed BERT-derived text must
reproduce the stored ones -- any mismatch would falsify the claim above.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .compare import load_checkpoint
from .data import EmotionDataset, Normalizer, load_pickle, to_device
from .metrics import NAMES
from .paths import ATT3, ATT4, LOGS, OUT, PREDICTIONS, ensure_dirs, selected_checkpoint_path
from .text_bert_encoder import encode, load_bert

BERT_DIR = OUT / "bert" / "bert-base-uncased"


def predict_folder(model, normalizer, device, folder: Path, with_diagnostics: bool):
    rows = []
    with torch.no_grad():
        for path in sorted(folder.glob("*.pkl")):
            raw = load_pickle(path)
            record = raw["test"] if (isinstance(raw, dict) and "test" in raw) else raw
            if not isinstance(record, dict):
                raise TypeError(f"Unexpected record type in {path}")
            record = dict(record)
            bert = np.asarray(record["text_bert"])
            if bert.ndim == 2:
                bert = bert[None]
            # --- the only change vs the student-path run: supply the 768-d slot
            record["text"] = encode(model_bert, torch.from_numpy(bert).long()).numpy()

            dataset = EmotionDataset(record, normalizer, labeled=False)
            for i in range(len(dataset)):
                single = bert[i]
                attention = single[1].astype(bool)
                attended = np.flatnonzero(attention)
                boundary_uncertain = bool(
                    not len(attended) or np.any(np.diff(attention.astype(int)) > 0)
                    or single[0, attended[-1]] != 102)
                batch = {key: value.unsqueeze(0) for key, value in dataset[i].items()}
                batch = to_device(batch, device)
                output = model(batch)
                probabilities = torch.softmax(output["logits"], dim=-1)[0].cpu().numpy()
                prediction = int(np.argmax(probabilities))
                row = {
                    "file": path.name,
                    "item_index": i,
                    "predicted_label": prediction,
                    "predicted_annotation": NAMES[prediction],
                    "predicted_strength": float(output["strength"][0].cpu()),
                    "p_negative": float(probabilities[0]),
                    "p_neutral": float(probabilities[1]),
                    "p_positive": float(probabilities[2]),
                }
                if with_diagnostics:
                    row.update({
                        "attention_valid_length": int(batch["valid"].sum().cpu()),
                        "content_length": int(batch["content"].sum().cpu()),
                        "tail_boundary_uncertain": boundary_uncertain,
                        "text_feature_source": "bert_base_uncased_on_text_bert",
                    })
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path,
                        default=LOGS / "model_selection_bertmask_20260924.json")
    parser.add_argument("--bert-dir", type=Path, default=BERT_DIR,
                        help="Directory with verified config.json and pytorch_model.bin")
    parser.add_argument("--output-tag", default="bertmask_20260924")
    parser.add_argument("--publish-canonical", action="store_true")
    args = parser.parse_args()
    ensure_dirs()

    global model_bert
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_bert = load_bert(args.bert_dir, device)
    decision = json.loads(args.selection.read_text(encoding="utf-8"))
    checkpoint = selected_checkpoint_path(decision)
    model, _ = load_checkpoint(checkpoint, device)
    model.eval()
    normalizer = Normalizer.load(OUT / "normalizer.npz")
    print(f"锁定模型 {checkpoint}  text_source={decision['text_source']}")

    # ---- attachment 4 control: BERT text must reproduce the stored predictions
    stored_path = PREDICTIONS / f"attachment4_predictions_explanations_{args.output_tag}.csv"
    rows4 = predict_folder(model, normalizer, device, ATT4, False)
    control = {"checked": 0, "label_mismatch": 0, "strength_max_abs_diff": 0.0}
    if stored_path.exists():
        with stored_path.open(encoding="utf-8-sig", newline="") as handle:
            stored = {r["file"]: r for r in csv.DictReader(handle)}
        for row in rows4:
            ref = stored.get(row["file"])
            if ref is None:
                continue
            control["checked"] += 1
            if int(float(ref["predicted_label"])) != row["predicted_label"]:
                control["label_mismatch"] += 1
            control["strength_max_abs_diff"] = max(
                control["strength_max_abs_diff"],
                abs(float(ref["predicted_strength"]) - row["predicted_strength"]))
    print(f"附件4 对照：核对 {control['checked']} 条，标签不一致 {control['label_mismatch']} 条，"
          f"强度最大绝对差 {control['strength_max_abs_diff']:.3e}")

    # ---- attachment 3: regenerate with the strong text path
    rows3 = predict_folder(model, normalizer, device, ATT3, True)
    out3 = PREDICTIONS / f"attachment3_predictions_berttext_{args.output_tag}.csv"
    if out3.exists():
        raise FileExistsError(out3)
    write_csv(out3, rows3)
    canonical3 = PREDICTIONS / "attachment3_predictions_20260924.csv"
    if args.publish_canonical:
        write_csv(canonical3, rows3)

    old3 = PREDICTIONS / "attachment3_predictions_studenttext_20260924.csv"
    changed = 0
    if old3.exists():
        with old3.open(encoding="utf-8-sig", newline="") as handle:
            old = {(r["file"], r["item_index"]): r for r in csv.DictReader(handle)}
        for row in rows3:
            ref = old.get((row["file"], str(row["item_index"])))
            if ref and int(float(ref["predicted_label"])) != row["predicted_label"]:
                changed += 1

    summary = {"rows": len(rows3), "csv": str(out3),
               "canonical_csv": str(canonical3) if args.publish_canonical else None,
               "attachment4_control": control,
               "attachment3_label_changes_vs_student_text_path": changed,
               "text_feature_source": "frozen bert-base-uncased on text_bert"}
    (LOGS / f"attachment3_berttext_prediction_{args.output_tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
