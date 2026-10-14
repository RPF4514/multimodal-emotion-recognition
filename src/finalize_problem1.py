"""Apply the supplied 21-row review without changing original samples or labels."""
from __future__ import annotations

import csv
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .paths import PROBLEM1
from .problem1 import (AUDIO_DIM, MAX_STEPS, SAMPLE_RATE, TEXT_DIM, VISION_DIM,
                       audio_features, decode_audio, extract_video_features,
                       find_ffmpeg, text_features, tokens)

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REVIEW_PATH = PROBLEM1 / "manual_review_21.xlsx"
AUDITED = PROBLEM1 / "audited_20260924"


def read_review() -> dict[tuple[str, str], dict]:
    """Read the existing xlsx using the standard library; never rewrite it."""
    with zipfile.ZipFile(REVIEW_PATH) as archive:
        shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = ["".join(node.itertext()) for node in shared.findall(f"{{{MAIN_NS}}}si")]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    records = []
    for row in sheet.findall(f".//{{{MAIN_NS}}}sheetData/{{{MAIN_NS}}}row"):
        number = int(row.attrib["r"])
        if number < 7:
            continue
        fields = [""] * 12
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            col = re.match(r"[A-Z]+", cell.attrib["r"]).group()
            index = 0
            for letter in col:
                index = index * 26 + ord(letter) - 64
            if index > len(fields):
                continue
            value = cell.find(f"{{{MAIN_NS}}}v")
            if value is None:
                continue
            fields[index - 1] = strings[int(value.text)] if cell.attrib.get("t") == "s" else value.text
        records.append(fields)
    header, body = records[0], records[1:]
    if len(body) != 21 or header[0:2] != ["video_id", "clip_id"]:
        raise ValueError("The supplied review workbook must have 21 identified rows.")
    review = {}
    for values in body:
        row = dict(zip(header, values))
        key = (row["video_id"], str(row["clip_id"]))
        if key in review:
            raise ValueError(f"Duplicate review key: {key}")
        review[key] = row
    return review


def independent_positions(duration: float) -> list[dict]:
    edges = np.linspace(0.0, duration, MAX_STEPS + 1)
    return [{"step": i, "start_second": round(float(edges[i]), 4),
             "end_second": round(float(edges[i + 1]), 4),
             "time_axis": "audio_video_only", "text": "",
             "alignment_method": "av_time_windows_text_sample"}
            for i in range(MAX_STEPS)]


def rebuild_reviewed(existing: dict, review: dict, ffmpeg: Path):
    duration = float(existing["original_duration_seconds"])
    if duration <= 0:
        raise ValueError(f"No valid video duration: {existing['id']}")
    source_words = tokens(existing["text"])
    text_words = source_words[:MAX_STEPS]
    text_array = np.zeros((MAX_STEPS, TEXT_DIM), np.float32)
    for i, word in enumerate(text_words):
        text_array[i] = text_features(word, i, max(1, len(text_words)))
    positions = independent_positions(duration)
    audio, rc, error = decode_audio(Path(existing["file"]), ffmpeg)
    audio_array = np.zeros((MAX_STEPS, AUDIO_DIM), np.float32)
    for i, row in enumerate(positions):
        start = max(0, min(len(audio), round(row["start_second"] * SAMPLE_RATE)))
        end = max(start, min(len(audio), round(row["end_second"] * SAMPLE_RATE)))
        # Extraction runs even on a quiet or silent track; a zero vector then
        # reflects the supplied signal instead of an invented missing label.
        audio_array[i] = audio_features(audio[start:end])
        row["audio_track_decoded"] = bool(rc == 0 and len(audio) > 0)
        row["speech_activity"] = review["speech_activity"]
    vision_array, video_ok, decoded_frames, reported_frames, fps = extract_video_features(
        Path(existing["file"]), positions, ffmpeg)
    for i, row in enumerate(positions):
        row["video_frame_decoded"] = bool(video_ok[i])
        row.setdefault("frame_number_zero_based", None)
    updated = dict(existing)
    updated.update({
        "previous_alignment_method": existing["alignment_method"],
        "alignment_method": "av_time_windows_text_sample",
        "alignment_strategy": review["alignment_strategy"],
        "review_status": "user_supplied_manual_observation",
        "text_av_correspondence": review["text_av_correspondence"],
        "word_level_alignable": review["word_level_alignable"],
        "speech_activity": review["speech_activity"],
        "speech_language": review["speech_language"],
        "audio_quality": review["audio_quality"],
        "review_note": review["review_note"],
        "reviewer_id": review["reviewer_id"],
        "review_date": review["review_date"],
        "valid_length": MAX_STEPS,
        "text_valid_length": len(text_words),
        "av_valid_length": MAX_STEPS,
        "text_was_truncated": len(source_words) > MAX_STEPS,
        "word_count": len(source_words),
        "text_sequence": text_words,
        "text_word_timestamps": [{"word": word, "start_second": None,
                                  "end_second": None} for word in text_words],
        "phrase_observation_about_second": 7.0 if
            (existing["video_id"], existing["clip_id"]) == ("-HwX2H8Z4hY", "2") else None,
        "positions": positions,
        "audio_decode_returncode": rc,
        "audio_decode_error": error,
        "audio_duration_seconds": round(len(audio) / SAMPLE_RATE, 4),
        "audio_rms": round(float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0, 6),
        "audio_track_decoded": bool(rc == 0 and len(audio) > 0),
        "audio_feature_extraction_performed": True,
        "decoded_video_positions": int(video_ok.sum()),
        "decoded_frames_until_last_requested": decoded_frames,
        "reported_frames": reported_frames,
        "fps": round(fps, 4),
    })
    return updated, text_array, audio_array, vision_array


def main():
    review = read_review()
    source = np.load(PROBLEM1 / "features_100.npz")
    ids = list(map(str, source["ids"]))
    original = [json.loads(line) for line in
                (PROBLEM1 / "time_map.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(ids) != 100 or len(original) != 100 or ids != [row["id"] for row in original]:
        raise ValueError("Existing 100-row feature and time-map order do not match.")
    found_keys = {(r["video_id"], str(r["clip_id"])) for r in original}
    if not set(review).issubset(found_keys):
        raise ValueError(f"Review keys missing from original data: {set(review) - found_keys}")
    ffmpeg = find_ffmpeg()
    text = source["text"].copy()
    audio = source["audio"].copy()
    vision = source["vision"].copy()
    valid_lengths = source["valid_lengths"].copy()
    text_lengths = valid_lengths.copy()
    av_lengths = valid_lengths.copy()
    revised = []
    for i, row in enumerate(original):
        key = (row["video_id"], str(row["clip_id"]))
        if key in review:
            updated, text[i], audio[i], vision[i] = rebuild_reviewed(row, review[key], ffmpeg)
            valid_lengths[i] = MAX_STEPS
            text_lengths[i] = updated["text_valid_length"]
            av_lengths[i] = MAX_STEPS
            print(f"rebuilt {i + 1}/100 {row['id']}", flush=True)
        else:
            updated = dict(row)
            updated.update({"review_status": "not_manually_reviewed",
                            "word_level_alignable": "automatic_asr_candidate",
                            "text_valid_length": int(valid_lengths[i]),
                            "av_valid_length": int(valid_lengths[i]),
                            "audio_track_decoded": bool(row["audio_decode_returncode"] == 0 and
                                                        row["audio_duration_seconds"] > 0),
                            "audio_feature_extraction_performed": True,
                            "alignment_strategy": "automatic_asr_word_anchors"})
        revised.append(updated)
    AUDITED.mkdir(parents=True, exist_ok=True)
    map_path = AUDITED / "time_map_100.jsonl"
    with map_path.open("w", encoding="utf-8") as handle:
        for row in revised:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    feature_path = AUDITED / "features_100.npz"
    np.savez_compressed(feature_path, ids=np.asarray(ids), valid_lengths=valid_lengths,
                        text_valid_lengths=text_lengths, av_valid_lengths=av_lengths,
                        text=text, audio=audio, vision=vision)
    coverage_fields = ["id", "video_id", "clip_id", "review_status", "text_av_correspondence",
                       "word_level_alignable", "speech_activity", "speech_language",
                       "audio_quality", "alignment_strategy", "previous_alignment_method",
                       "alignment_method", "review_note", "reviewer_id", "review_date",
                       "original_duration_seconds", "text_valid_length", "av_valid_length",
                       "audio_track_decoded", "audio_feature_extraction_performed",
                       "decoded_video_positions"]
    coverage_path = AUDITED / "coverage_100.csv"
    with coverage_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=coverage_fields)
        writer.writeheader()
        writer.writerows([{name: row.get(name, "") for name in coverage_fields} for row in revised])
    counts = {status: sum(row["review_status"] == status for row in revised)
              for status in ("user_supplied_manual_observation", "not_manually_reviewed")}
    report = {"source_review": str(REVIEW_PATH), "source_features": str(PROBLEM1 / "features_100.npz"),
              "source_time_map": str(PROBLEM1 / "time_map.jsonl"),
              "outputs": {"features": str(feature_path), "time_map": str(map_path),
                          "coverage": str(coverage_path)},
              "sample_count": len(revised), "unique_ids": len(set(ids)), "review_status_counts": counts,
              "reviewed_word_seconds_nonnull": sum(
                  timestamp["start_second"] is not None for row in revised if
                  row["review_status"] == "user_supplied_manual_observation"
                  for timestamp in row["text_word_timestamps"]),
              "reviewed_video_frames_decoded": sum(row["decoded_video_positions"]
                                                   for row in revised if row["review_status"] != "not_manually_reviewed"),
              "reviewed_audio_tracks_decoded": sum(row["audio_track_decoded"]
                                                   for row in revised if row["review_status"] != "not_manually_reviewed"),
              "text_truncated_samples": [row["id"] for row in revised if row.get("text_was_truncated")],
              "policy": "Reviewed text and AV are separate sequences; no reviewed word timestamps are inferred."}
    (AUDITED / "run_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
