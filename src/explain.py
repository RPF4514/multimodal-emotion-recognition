from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .compare import load_checkpoint
from .data import EmotionDataset, Normalizer, load_pickle, reencode_masked_text, to_device
from .metrics import NAMES
from .paths import ATT4, FIGURES, LOGS, OUT, PREDICTIONS, PROBLEM1, PROJECT, ensure_dirs, selected_checkpoint_path
from .problem1 import find_ffmpeg
from .text_bert_encoder import encode, load_bert

MODES = ("text", "audio", "vision")


def remove_mode(batch, mode, start, end, bert_encoder=None):
    new = {key: value.clone() for key, value in batch.items()}
    full_modality = start == 0 and end >= new["valid"].shape[1]
    target = new["valid"][:, start:end] if full_modality else new["content"][:, start:end]
    new["observed"][:, start:end, mode] &= ~target
    if mode == 0:
        new["text_mask"][:, start:end] &= ~target
        new["token_ids"][:, start:end] = torch.where(
            target, 0, new["token_ids"][:, start:end])
        if "teacher_text" in new:
            if full_modality:
                new["teacher_text"].zero_()
            else:
                new = reencode_masked_text(new, bert_encoder)
    elif mode == 1:
        new["audio"][:, start:end] *= (~target).unsqueeze(-1)
    else:
        new["vision"][:, start:end] *= (~target).unsqueeze(-1)
    return new


def explain_one(model, batch, raw_text, window=4, bert_encoder=None):
    with torch.no_grad():
        full = model(batch)
        p = torch.softmax(full["logits"], dim=-1)[0]
        selected = int(p.argmax())
        length = int(batch["valid"][0].sum())
        base_prob = float(p[selected])
        base_strength = float(full["strength"][0])
        mode_drops, mode_reg = [], []
        for mode in range(3):
            changed = model(remove_mode(batch, mode, 0, batch["valid"].shape[1], bert_encoder))
            changed_prob = torch.softmax(changed["logits"], dim=-1)[0, selected]
            mode_drops.append(float(p[selected] - changed_prob))
            mode_reg.append(abs(base_strength - float(changed["strength"][0])))
        positive = np.maximum(mode_drops, 0)
        shares = positive / positive.sum() if positive.sum() > 1e-8 else np.zeros(3)
        local = []
        for mode in range(3):
            for start in range(length):
                if not bool(batch["content"][0, start] and batch["observed"][0, start, mode]):
                    continue
                end = min(length, start + window)
                changed = model(remove_mode(batch, mode, start, end, bert_encoder))
                changed_prob = torch.softmax(changed["logits"], dim=-1)[0, selected]
                local.append({"mode": MODES[mode], "start_step": start,
                              "end_step_exclusive": end,
                              "class_probability_drop": float(p[selected] - changed_prob),
                              "strength_change_abs": abs(base_strength - float(changed["strength"][0]))})
        local.sort(key=lambda row: row["class_probability_drop"], reverse=True)
        best = local[0] if local else None
        return {
            "occlusion_protocol": "prebert_token_mask_reencode_context",
            "predicted_label": selected, "predicted_annotation": NAMES[selected],
            "predicted_strength": base_strength, "predicted_class_probability": base_prob,
            "p_negative": float(p[0]), "p_neutral": float(p[1]), "p_positive": float(p[2]),
            "valid_length": length, "text_contribution_signed": mode_drops[0],
            "audio_contribution_signed": mode_drops[1], "vision_contribution_signed": mode_drops[2],
            "text_share_positive": float(shares[0]), "audio_share_positive": float(shares[1]),
            "vision_share_positive": float(shares[2]),
            "primary_mode": MODES[int(np.argmax(shares))] if shares.sum() > 0 else "none",
            "text_strength_change_abs": mode_reg[0], "audio_strength_change_abs": mode_reg[1],
            "vision_strength_change_abs": mode_reg[2],
            "key_mode": best["mode"] if best else "none",
            "key_start_step": best["start_step"] if best else "",
            "key_end_step_exclusive": best["end_step_exclusive"] if best else "",
            "key_probability_drop": best["class_probability_drop"] if best else "",
            "all_window_scores": local,
        }


def plot_examples(rows, output_tag):
    for row in rows[:3]:
        values = [row[f"{mode}_contribution_signed"] for mode in MODES]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(MODES, values, color=["#4F81BD", "#9BBB59", "#C0504D"])
        ax.axhline(0, color="black", lw=0.7)
        ax.set_ylabel("Predicted class probability drop")
        ax.set_title(f"Sample {row['file']} modality contributions")
        fig.tight_layout()
        fig.savefig(FIGURES / f"explain_{output_tag}_{row['file'].replace('.pkl','')}.png", dpi=160)
        plt.close(fig)


def locate_video_evidence(path, result, time_map, ffmpeg, output_tag):
    """Map model steps through verified BERT offsets and ASR word times."""
    mapping = time_map.get(path.name)
    empty = {"evidence_text": "", "evidence_start_second": "",
             "evidence_end_second": "", "evidence_frame_number": "",
             "evidence_alignment_method": "unavailable", "bert_text_match": False,
             "evidence_alignment_confidence": "unavailable",
             "evidence_text_role": "unavailable",
             "key_frame_image": "", "key_frame_status": "time_map_or_window_unavailable"}
    if mapping is None or result["key_start_step"] == "":
        return empty
    positions = mapping["positions"][result["key_start_step"]:result["key_end_step_exclusive"]]
    with_text = [p for p in positions if p["character_start"] is not None]
    timed = [p for p in positions if p["start_second"] is not None]
    if not timed:
        return {**empty, "key_frame_status": "window_has_only_special_tokens",
                "bert_text_match": mapping["bert_prefix_matches"]}
    first = min(p["character_start"] for p in with_text) if with_text else 0
    last = max(p["character_end"] for p in with_text) if with_text else 0
    excerpt = mapping["raw_text"][first:last]
    start = min(p["start_second"] for p in timed)
    end = max(p["end_second"] for p in timed)
    frame_number = round((start + end) / 2 * mapping["video_fps"])
    image_path = ""
    status = "target_frame_not_decodable"
    video = Path(mapping["video"])
    if not video.is_file():
        video = ATT4 / "videos" / video.name
    if video.exists():
        directory = FIGURES / f"keyframes_aligned_{output_tag}"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{path.stem}.jpg"
        command = [str(ffmpeg), "-loglevel", "error", "-ss", str((start + end) / 2),
                   "-i", str(video), "-frames:v", "1", "-y", str(target)]
        completed = subprocess.run(command, capture_output=True)
        if completed.returncode == 0 and target.exists() and target.stat().st_size > 0:
            image_path, status = target.relative_to(PROJECT).as_posix(), "ok"
    else:
        status = "video_missing"
    return {"evidence_text": excerpt, "evidence_start_second": round(start, 3),
            "evidence_end_second": round(end, 3), "evidence_frame_number": frame_number,
            "evidence_alignment_method": mapping["alignment_method"],
            "evidence_alignment_confidence": (
                "low_energy_estimate" if "energy" in mapping["alignment_method"]
                else "automatic_asr_estimate"),
            "evidence_text_role": ("text_evidence" if result["key_mode"] == "text"
                                   else "simultaneous_text_context"),
            "bert_text_match": mapping["bert_prefix_matches"],
            "key_frame_image": image_path, "key_frame_status": status}


def faithfulness_summary(detail, output_tag):
    top, bottom = [], []
    for result in detail.values():
        scores = [r["class_probability_drop"] for r in result["all_window_scores"]]
        if scores:
            top.append(max(scores))
            bottom.append(min(scores))
    summary = {"top_window_mean_probability_drop": float(np.mean(top)),
               "bottom_window_mean_probability_drop": float(np.mean(bottom)),
               "top_minus_bottom": float(np.mean(np.asarray(top) - np.asarray(bottom))),
               "sample_count": len(top),
               "note": "Post-hoc comparison on selected windows; not a causal claim or held-out explanation test."}
    (LOGS / f"explanation_faithfulness_{output_tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path,
                        default=LOGS / "model_selection_bertmask_20260924.json")
    parser.add_argument("--bert-dir", type=Path,
                        default=OUT / "bert" / "bert-base-uncased")
    parser.add_argument("--output-tag", default="bertmask_20260924")
    args = parser.parse_args()
    ensure_dirs()
    torch.set_num_threads(4)
    decision = json.loads(args.selection.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(selected_checkpoint_path(decision), device)
    bert_encoder = load_bert(args.bert_dir, device)
    normalizer = Normalizer.load(OUT / "normalizer.npz")
    time_map = {row["file"]: row for row in
                (json.loads(line) for line in (PROBLEM1 / "attachment4_time_map.jsonl")
                 .read_text(encoding="utf-8").splitlines())}
    ffmpeg = find_ffmpeg()
    rows, detail = [], {}
    for path in sorted(ATT4.glob("*.pkl")):
        raw = load_pickle(path)
        # Attachment 4 uses the same frozen BERT text interface as attachment
        # 2/3; do not silently switch a bridge model to its student fallback.
        if "text" not in raw:
            tokens = np.asarray(raw["text_bert"])
            if tokens.ndim == 2:
                tokens = tokens[None]
            raw = {**raw, "text": encode(bert_encoder, torch.from_numpy(tokens).long())[0].numpy()}
        dataset = EmotionDataset(raw, normalizer, labeled=False)
        batch = {key: value.unsqueeze(0) for key, value in dataset[0].items()}
        batch = to_device(batch, device)
        result = explain_one(model, batch, raw.get("raw_text", ""), bert_encoder=bert_encoder)
        result.update(locate_video_evidence(path, result, time_map, ffmpeg, args.output_tag))
        detail[path.stem] = result
        flat = {key: value for key, value in result.items() if key != "all_window_scores"}
        rows.append({"file": path.name, "id": raw.get("id", path.stem), **flat})
        print(f"explained {path.name}", flush=True)
    with (PREDICTIONS / f"attachment4_predictions_explanations_{args.output_tag}.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (LOGS / f"attachment4_window_explanations_{args.output_tag}.json").write_text(
        json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_examples(rows, args.output_tag)
    faithfulness = faithfulness_summary(detail, args.output_tag)
    print(json.dumps({"rows": len(rows), "faithfulness": faithfulness,
                      "csv": str(PREDICTIONS / f"attachment4_predictions_explanations_{args.output_tag}.csv")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
