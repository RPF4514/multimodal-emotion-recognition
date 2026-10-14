"""Map Attachment 4 BERT positions to original text and audio/video times."""
from __future__ import annotations

import json

import cv2
import numpy as np
from tokenizers import BertWordPieceTokenizer
from faster_whisper import WhisperModel

from .data import load_pickle
from .paths import ATT4, PROBLEM1, ensure_dirs
from .problem1 import (SAMPLE_RATE, TOKEN_RE, audio_activity, decode_audio,
                       find_ffmpeg, whisper_word_intervals, word_intervals)


def make_record(path, tokenizer, aligner, ffmpeg):
    raw = load_pickle(path)
    transcript = str(raw["raw_text"])
    bert = np.asarray(raw["text_bert"])
    valid = int(bert[1].sum())
    encoding = tokenizer.encode(transcript)
    same_prefix = encoding.ids[:valid - 1] == bert[0, :valid - 1].tolist()
    special_end = int(bert[0, valid - 1]) == 102
    if not (same_prefix and special_end):
        raise ValueError(f"BERT tokenization does not match supplied IDs: {path.name}")
    video = ATT4 / "videos" / f"{path.stem}.mp4"
    audio, rc, error = decode_audio(video, ffmpeg)
    duration = len(audio) / SAMPLE_RATE
    audio_rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    energy, start, end = audio_activity(audio, duration)
    aligned, match_rate, asr_count = None, 0.0, 0
    if audio_rms >= 0.001:
        try:
            aligned, match_rate, asr_count = whisper_word_intervals(audio, transcript, aligner)
        except Exception as exc:
            error = f"{error}; {exc}"[-300:]
    method = "whisper_tiny_en_word_anchor" if aligned is not None else "energy_guided_estimate"
    if aligned is None:
        from .problem1 import tokens
        aligned = word_intervals(tokens(transcript), energy, start, end)
    aligned = [(word, max(0.0, min(duration, a)), max(0.0, min(duration, b)))
               for word, a, b in aligned]
    spans = [(match.start(), match.end()) for match in TOKEN_RE.finditer(transcript)]
    if len(spans) != len(aligned):
        raise ValueError(f"Word/token span mismatch: {path.name}")
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    positions = []
    for step in range(valid):
        a, b = encoding.offsets[step] if step < len(encoding.offsets) else (0, 0)
        token = encoding.tokens[step] if step < len(encoding.tokens) else ""
        if step == valid - 1:
            token, a, b = "[SEP]", 0, 0
        hits = [i for i, (left, right) in enumerate(spans) if min(b, right) > max(a, left)]
        if hits and audio_rms >= 0.001:
            word_start = min(aligned[i][1] for i in hits)
            word_end = max(aligned[i][2] for i in hits)
            source_word = " ".join(aligned[i][0] for i in hits)
            midpoint = (word_start + word_end) / 2
            frame_index = min(max(0, frames - 1), round(midpoint * fps)) if fps > 0 else None
        else:
            word_start = word_end = frame_index = None
            source_word = " ".join(aligned[i][0] for i in hits) if hits else None
        positions.append({"step": step, "bert_token": token,
                          "character_start": a if hits else None,
                          "character_end": b if hits else None,
                          "source_word": source_word,
                          "start_second": round(word_start, 4) if hits else None,
                          "end_second": round(word_end, 4) if hits else None,
                          "frame_number_zero_based": frame_index,
                          "alignment_method": method})
    return {"file": path.name, "id": str(raw.get("id", path.stem)), "video": str(video),
            "raw_text": transcript, "valid_length": valid,
            "bert_prefix_matches": same_prefix, "bert_truncated": len(encoding.ids) > valid,
            "audio_duration_seconds": round(duration, 4), "audio_rms": round(audio_rms, 6),
            "audio_decode_returncode": rc, "audio_decode_error": error,
            "asr_reference_token_match_rate": round(match_rate, 4),
            "asr_word_count": asr_count, "alignment_method": method,
            "video_fps": round(fps, 4), "video_reported_frames": frames,
            "positions": positions}


def main():
    ensure_dirs()
    ffmpeg = find_ffmpeg()
    vocabulary = PROBLEM1 / "tokenizer" / "vocab.txt"
    if not vocabulary.is_file():
        raise FileNotFoundError(vocabulary)
    tokenizer = BertWordPieceTokenizer(str(vocabulary), lowercase=True)
    aligner = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=4,
                           download_root=str(PROBLEM1 / "pretrained"), local_files_only=True)
    records = []
    for path in sorted(ATT4.glob("*.pkl")):
        record = make_record(path, tokenizer, aligner, ffmpeg)
        records.append(record)
        print(f"{path.name} {record['alignment_method']} "
              f"match={record['asr_reference_token_match_rate']:.2f}", flush=True)
    output = PROBLEM1 / "attachment4_time_map.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"files": len(records), "bert_exact_prefix": sum(r["bert_prefix_matches"] for r in records),
               "bert_truncated": sum(r["bert_truncated"] for r in records),
               "whisper_aligned": sum(r["alignment_method"].startswith("whisper") for r in records),
               "silent_audio": sum(r["audio_rms"] < 0.001 for r in records),
               "word_time_method": "ASR word anchor with gap interpolation, fallback energy-guided estimate"}
    (PROBLEM1 / "attachment4_alignment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
