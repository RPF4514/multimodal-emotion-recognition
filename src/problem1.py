"""Extract auditable features from all 100 supplied videos.

The transcript timestamps are energy-guided estimates, not forced alignment.
Every position keeps its source time and a declared alignment method.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .paths import ATT1, FIGURES, PROBLEM1, ensure_dirs

NAMESPACE = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*[,.!?;:]?|\d+", re.UNICODE)
SAMPLE_RATE = 16000
MAX_STEPS = 50
TEXT_DIM = 32
AUDIO_DIM = 74
VISION_DIM = 35


def label_rows(path: Path):
    """Read the five-column workbook using only Python's standard library."""
    with zipfile.ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = ["".join(x.itertext()) for x in shared_root.findall("s:si", NAMESPACE)]
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in root.findall(".//s:sheetData/s:row", NAMESPACE):
        cells = []
        for cell in row.findall("s:c", NAMESPACE):
            value = cell.find("s:v", NAMESPACE)
            if value is None:
                cells.append("")
            elif cell.get("t") == "s":
                cells.append(strings[int(value.text)])
            else:
                cells.append(value.text or "")
        if cells:
            rows.append(cells)
    assert rows[0][:5] == ["video_id", "clip_id", "text", "label", "annotation"]
    return [dict(zip(rows[0][:5], row[:5])) for row in rows[1:]]


def find_ffmpeg(explicit: str | None = None) -> Path:
    candidates = [explicit, shutil.which("ffmpeg"),
                  str(Path.home() / "AppData" / "Roaming" / "bilibili" / "ffmpeg" / "ffmpeg.exe")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("FFmpeg is required for MP4 audio extraction; pass --ffmpeg PATH")


def decode_audio(video: Path, ffmpeg: Path):
    command = [str(ffmpeg), "-v", "error", "-i", str(video), "-vn", "-ac", "1",
               "-ar", str(SAMPLE_RATE), "-f", "s16le", "pipe:1"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=False, timeout=120)
    pcm = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768
    return pcm, result.returncode, result.stderr.decode("utf-8", errors="replace")[-300:]


def tokens(text: str):
    return TOKEN_RE.findall(text)


def syllable_weight(token: str):
    groups = re.findall(r"[aeiouy]+", token.lower())
    return max(1, len(groups)) + (0.6 if token in ".!?;:" else 0.0)


def audio_activity(audio: np.ndarray, duration: float):
    hop = 320  # 20 ms
    if len(audio) == 0:
        return np.zeros(max(1, int(duration / 0.02)), dtype=np.float32), 0.0, duration
    padded = np.pad(audio, (0, (-len(audio)) % hop))
    energy = np.sqrt(np.mean(padded.reshape(-1, hop) ** 2, axis=1))
    if not np.any(energy > 1e-5):
        return energy, 0.0, duration
    threshold = max(0.004, float(np.percentile(energy, 20)) * 2.0)
    active = np.flatnonzero(energy >= threshold)
    if len(active) == 0:
        return energy, 0.0, duration
    return energy, max(0.0, active[0] * 0.02 - 0.08), min(duration, (active[-1] + 1) * 0.02 + 0.08)


def word_intervals(words: list[str], energy: np.ndarray, start: float, end: float):
    """Constrain weighted word boundaries to local acoustic-energy minima."""
    if not words:
        return []
    weights = np.asarray([syllable_weight(word) for word in words], dtype=float)
    fractions = np.cumsum(weights)[:-1] / weights.sum()
    boundaries = [start]
    for fraction in fractions:
        target = start + float(fraction) * (end - start)
        left = max(0, int((target - 0.12) / 0.02))
        right = min(len(energy), int((target + 0.12) / 0.02) + 1)
        if right > left:
            local = left + int(np.argmin(energy[left:right]))
            target = 0.7 * target + 0.3 * (local * 0.02)
        boundaries.append(max(boundaries[-1], min(end, target)))
    boundaries.append(end)
    return [(word, float(boundaries[i]), float(boundaries[i + 1]))
            for i, word in enumerate(words)]


def align_to_asr(reference: list[str], asr_words: list[tuple[str, float, float]]):
    """Anchor supplied transcript to ASR word timestamps, interpolating gaps."""
    normalized = lambda word: re.sub(r"[^a-z0-9]", "", word.lower())
    left = [normalized(word) for word in reference]
    right = [normalized(word) for word, _, _ in asr_words]
    anchors = {}
    for block in difflib.SequenceMatcher(None, left, right, autojunk=False).get_matching_blocks():
        for offset in range(block.size):
            anchors[block.a + offset] = asr_words[block.b + offset][1:]
    match_rate = len(anchors) / max(1, len(reference))
    if not anchors or match_rate < 0.60:
        return None, match_rate
    intervals: list[tuple[str, float, float] | None] = [None] * len(reference)
    for index, (start, end) in anchors.items():
        intervals[index] = (reference[index], float(start), float(end))
    indices = sorted(anchors)
    for begin, finish in zip([-1, *indices], [*indices, len(reference)]):
        missing = list(range(begin + 1, finish))
        if not missing:
            continue
        if begin < 0:
            right_edge = intervals[finish][1]
            left_edge = max(0.0, right_edge - 0.24 * len(missing))
        elif finish == len(reference):
            left_edge = intervals[begin][2]
            right_edge = left_edge + 0.24 * len(missing)
        else:
            left_edge, right_edge = intervals[begin][2], intervals[finish][1]
        right_edge = max(left_edge, right_edge)
        weights = np.asarray([syllable_weight(reference[i]) for i in missing], dtype=float)
        edges = left_edge + np.r_[0, np.cumsum(weights) / weights.sum()] * (right_edge - left_edge)
        for j, index in enumerate(missing):
            intervals[index] = (reference[index], float(edges[j]), float(edges[j + 1]))
    return intervals, match_rate


def whisper_word_intervals(audio: np.ndarray, text: str, model):
    words = tokens(text)
    segments, _ = model.transcribe(audio, language="en", word_timestamps=True,
                                   beam_size=1, condition_on_previous_text=False,
                                   vad_filter=False)
    predicted = [(word.word.strip(), float(word.start), float(word.end))
                 for segment in segments for word in (segment.words or [])
                 if re.search(r"[A-Za-z0-9]", word.word)]
    aligned, match_rate = align_to_asr(words, predicted)
    return aligned, match_rate, len(predicted)


def position_intervals(words: list[tuple[str, float, float]], duration: float, positions=MAX_STEPS):
    count = min(positions, max(1, len(words)))
    groups = np.array_split(np.arange(len(words)), count) if words else [np.array([], dtype=int)]
    result = []
    for step, group in enumerate(groups):
        if len(group):
            a, b = words[int(group[0])][1], words[int(group[-1])][2]
            content = " ".join(words[int(j)][0] for j in group)
        else:
            a, b, content = 0.0, duration, ""
        result.append({"step": step, "start_second": round(a, 4),
                       "end_second": round(b, 4), "text": content,
                       "alignment_method": "energy_guided_estimate"})
    return result


def text_features(text: str, position: int, count: int):
    vec = np.zeros(TEXT_DIM, dtype=np.float32)
    words = tokens(text)
    if not words:
        return vec
    joined = " ".join(words).lower()
    vec[:8] = [min(len(joined) / 100, 1), min(len(words) / 10, 1),
               sum(c.isupper() for c in text) / max(1, len(text)),
               text.count("!") / max(1, len(words)),
               text.count("?") / max(1, len(words)),
               position / max(1, count - 1),
               float(any(w.lower() in {"not", "never", "no", "n't"} for w in words)),
               float(any(w.lower() in {"very", "really", "extremely"} for w in words))]
    # Reproducible character-trigram counts; this is a weak lexical baseline,
    # not a pre-trained language embedding or a claimed sentiment score.
    for i in range(max(1, len(joined) - 2)):
        gram = joined[i:i + 3].encode("utf-8")
        bucket = int.from_bytes(gram[:3].ljust(3, b"\0"), "little") % 24
        vec[8 + bucket] += 1
    vec[8:] /= max(1, len(joined) - 2)
    return vec


def mel_bank():
    hz_to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    mel_to_hz = lambda mel: 700 * (10 ** (mel / 2595) - 1)
    mel = np.linspace(hz_to_mel(50), hz_to_mel(7600), 66)
    bins = np.floor((512 + 1) * mel_to_hz(mel) / SAMPLE_RATE).astype(int)
    bank = np.zeros((64, 257), dtype=np.float32)
    for i in range(64):
        a, b, c = np.clip(bins[i:i + 3], 0, 256)
        if a < b:
            bank[i, a:b] = np.linspace(0, 1, b - a, endpoint=False)
        if b < c:
            bank[i, b:c] = np.linspace(1, 0, c - b, endpoint=False)
    return bank


MEL_BANK = mel_bank()


def audio_features(chunk: np.ndarray):
    out = np.zeros(AUDIO_DIM, dtype=np.float32)
    if len(chunk) < 400:
        return out
    # Analyze up to 80 evenly spaced 25-ms frames per aligned interval.
    starts = np.linspace(0, len(chunk) - 400, min(80, max(1, len(chunk) // 160))).astype(int)
    windows = np.stack([chunk[s:s + 400] for s in starts]) * np.hanning(400)
    spectrum = np.abs(np.fft.rfft(windows, n=512)).astype(np.float32)
    power = spectrum ** 2
    mel = np.log1p(power @ MEL_BANK.T)
    out[:64] = mel.mean(0)
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    freqs = np.fft.rfftfreq(512, 1 / SAMPLE_RATE)
    mass = spectrum.mean(0)
    total_mass = float(mass.sum())
    if total_mass <= 1e-12:
        # Spectral centroid and roll-off are undefined for an exactly silent
        # interval; record zeros instead of a false mid-band frequency.
        centroid, rolloff = 0.0, 0.0
    else:
        centroid = float((mass * freqs).sum() / total_mass)
        rolloff_index = min(len(freqs) - 1,
                            int(np.searchsorted(np.cumsum(mass), total_mass * 0.85)))
        rolloff = float(freqs[rolloff_index])
    zcr = float(np.mean(np.signbit(chunk[1:]) != np.signbit(chunk[:-1])))
    spectral_flux = float(np.mean(np.abs(np.diff(spectrum, axis=0)))) if len(starts) > 1 else 0
    out[64:] = [rms, float(np.max(np.abs(chunk))), zcr, centroid / 8000,
                rolloff / 8000, float(np.std(chunk)), float(np.mean(chunk)),
                spectral_flux / 100, float(np.mean(mel.std(axis=0))),
                float(np.mean(np.abs(np.diff(chunk))))]
    return out


def frame_features(frame: np.ndarray, previous: np.ndarray | None):
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    out = np.zeros(VISION_DIM, dtype=np.float32)
    out[:3] = rgb.mean((0, 1))
    out[3:6] = rgb.std((0, 1))
    out[6:22], _ = np.histogram(gray, bins=16, range=(0, 1), density=True)
    out[6:22] /= 16
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    orientation = (np.arctan2(gy, gx) + np.pi) % np.pi
    out[22:30], _ = np.histogram(orientation, bins=8, range=(0, np.pi), weights=magnitude)
    out[22:30] /= max(1e-8, out[22:30].sum())
    out[30] = float(magnitude.mean())
    out[31] = float(np.mean(np.abs(gray - previous))) if previous is not None else 0
    out[32] = float(gray.std())
    out[33] = float(np.mean(gray > 0.75))
    out[34] = float(np.mean(gray < 0.25))
    return out, gray


def extract_video_features(video: Path, positions: list[dict], ffmpeg: Path):
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    features = np.zeros((MAX_STEPS, VISION_DIM), dtype=np.float32)
    success = np.zeros(MAX_STEPS, dtype=bool)
    if fps <= 0:
        return features, success, 0, reported_frames, fps
    # FFmpeg can decode later frames of supplied MP4s that OpenCV stops reading early.
    command = [str(ffmpeg), "-v", "error", "-i", str(video), "-an", "-vf",
               "scale=160:90", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    decoded = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             check=False, timeout=120)
    frame_bytes = 160 * 90 * 3
    actual_frames = len(decoded.stdout) // frame_bytes
    if actual_frames == 0:
        return features, success, 0, reported_frames, fps
    frames = np.frombuffer(decoded.stdout[:actual_frames * frame_bytes], np.uint8).reshape(
        actual_frames, 90, 160, 3)
    targets = [max(0, round((row["start_second"] + row["end_second"]) / 2 * fps))
               for row in positions]
    previous = None
    for step, target in enumerate(targets):
        if target >= actual_frames:
            continue
        features[step], previous = frame_features(frames[target], previous)
        success[step] = True
        positions[step]["frame_number_zero_based"] = target
    return features, success, actual_frames, reported_frames, fps


def process_one(label: dict, ffmpeg: Path, aligner=None, fallback_aligner=None):
    video_id, clip_id = label["video_id"], str(label["clip_id"])
    video = ATT1 / video_id / f"{clip_id}.mp4"
    if not video.exists():
        raise FileNotFoundError(video)
    audio, audio_rc, audio_error = decode_audio(video, ffmpeg)
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    video_duration = frames / fps if fps > 0 else 0
    audio_duration = len(audio) / SAMPLE_RATE
    audio_rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    audio_has_signal = audio_rms >= 0.001
    duration = audio_duration if audio_duration > 0 else video_duration
    if duration <= 0:
        raise ValueError(f"No usable duration for {video}")
    energy, active_start, active_end = audio_activity(audio, duration)
    aligned_words = None
    match_rate, asr_word_count = 0.0, 0
    if aligner is not None and audio_has_signal:
        try:
            aligned_words, match_rate, asr_word_count = whisper_word_intervals(
                audio, label["text"], aligner)
        except Exception as exc:
            audio_error = (audio_error + f"; word-alignment error: {exc}")[-300:]
    alignment_method = "whisper_tiny_en_asr_word_anchors_with_gap_interpolation"
    if aligned_words is None and fallback_aligner is not None and audio_has_signal:
        try:
            base_words, base_match, base_count = whisper_word_intervals(
                audio, label["text"], fallback_aligner)
            if base_match > match_rate:
                aligned_words, match_rate, asr_word_count = base_words, base_match, base_count
                alignment_method = "whisper_base_en_asr_word_anchors_with_gap_interpolation"
        except Exception as exc:
            audio_error = (audio_error + f"; base word-alignment error: {exc}")[-300:]
    if aligned_words is None:
        aligned_words = word_intervals(tokens(label["text"]), energy, active_start, active_end)
        alignment_method = ("no_audio_uniform_estimate" if not audio_has_signal else
                            "energy_guided_estimate_not_forced_alignment")
    aligned_words = [(word, max(0.0, min(duration, start)),
                      max(0.0, min(duration, end))) for word, start, end in aligned_words]
    positions = position_intervals(aligned_words, duration)
    for row in positions:
        row["alignment_method"] = alignment_method
    text_array = np.zeros((MAX_STEPS, TEXT_DIM), np.float32)
    audio_array = np.zeros((MAX_STEPS, AUDIO_DIM), np.float32)
    for i, row in enumerate(positions):
        text_array[i] = text_features(row["text"], i, len(positions))
        a = max(0, min(len(audio), round(row["start_second"] * SAMPLE_RATE)))
        b = max(a, min(len(audio), round(row["end_second"] * SAMPLE_RATE)))
        if audio_has_signal:
            audio_array[i] = audio_features(audio[a:b])
    vision_array, video_ok, decoded_frames, reported_frames, fps = extract_video_features(video, positions, ffmpeg)
    for i, row in enumerate(positions):
        row["audio_available"] = audio_has_signal
        row["video_frame_decoded"] = bool(video_ok[i])
        row.setdefault("frame_number_zero_based", "")
    result = {
        "id": f"{video_id}$_${clip_id}", "video_id": video_id, "clip_id": clip_id,
        "file": str(video), "text": label["text"],
        "original_duration_seconds": round(video_duration, 4),
        "audio_duration_seconds": round(audio_duration, 4),
        "audio_rms": round(audio_rms, 6),
        "audio_has_signal": audio_has_signal,
        "valid_length": len(positions), "word_count": len(aligned_words),
        "fps": round(fps, 4), "reported_frames": reported_frames,
        "decoded_frames_until_last_requested": decoded_frames,
        "audio_decode_returncode": audio_rc, "audio_decode_error": audio_error,
        "decoded_video_positions": int(video_ok.sum()),
        "video_decode_method": "ffmpeg_scaled_rawvideo",
        "alignment_method": alignment_method,
        "asr_reference_token_match_rate": round(match_rate, 4),
        "asr_word_count": asr_word_count,
        "positions": positions,
    }
    return result, text_array, audio_array, vision_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--alignment", choices=("energy", "whisper"), default="whisper")
    args = parser.parse_args()
    ensure_dirs()
    ffmpeg = find_ffmpeg(args.ffmpeg)
    labels = label_rows(ATT1 / "label-100.xlsx")
    videos = list(ATT1.rglob("*.mp4"))
    if len(labels) != 100 or len(videos) != 100:
        raise RuntimeError(f"Expected 100 labels and videos, got {len(labels)} / {len(videos)}")
    ffmpeg_version = subprocess.run([str(ffmpeg), "-version"], capture_output=True,
                                    text=True, errors="replace").stdout.splitlines()[0]
    aligner = None
    fallback_aligner = None
    if args.alignment == "whisper":
        from faster_whisper import WhisperModel
        aligner = WhisperModel("tiny.en", device="cpu", compute_type="int8", cpu_threads=4,
                               download_root=str(PROBLEM1 / "pretrained"), local_files_only=True)
        fallback_aligner = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=4,
                                        download_root=str(PROBLEM1 / "pretrained"), local_files_only=True)
    rows, text_list, audio_list, vision_list = [], [], [], []
    selected = labels[:args.limit] if args.limit else labels
    started = time.perf_counter()
    with (PROBLEM1 / "time_map.jsonl").open("w", encoding="utf-8") as handle:
        for index, label in enumerate(selected, 1):
            result, text_array, audio_array, vision_array = process_one(
                label, ffmpeg, aligner, fallback_aligner)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            text_list.append(text_array)
            audio_list.append(audio_array)
            vision_list.append(vision_array)
            rows.append({key: result[key] for key in
                         ("id", "video_id", "clip_id", "original_duration_seconds",
                          "audio_duration_seconds", "valid_length", "word_count", "fps",
                          "reported_frames", "decoded_frames_until_last_requested",
                          "decoded_video_positions", "audio_decode_returncode",
                          "audio_rms", "audio_has_signal", "alignment_method",
                          "asr_reference_token_match_rate", "asr_word_count")})
            print(f"{index}/{len(selected)} {result['id']} valid={result['valid_length']} "
                  f"video={result['decoded_video_positions']}", flush=True)
    with (PROBLEM1 / "coverage_100.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(PROBLEM1 / "features_100.npz",
                        ids=np.asarray([row["id"] for row in rows]),
                        valid_lengths=np.asarray([row["valid_length"] for row in rows]),
                        text=np.stack(text_list), audio=np.stack(audio_list), vision=np.stack(vision_list))
    summary = {"label_count": len(labels), "video_count": len(videos),
               "processed": len(rows), "all_ids_unique": len({r["id"] for r in rows}) == len(rows),
               "all_video_positions_decoded": sum(r["decoded_video_positions"] == r["valid_length"] for r in rows),
               "audio_decode_failures": sum(r["audio_decode_returncode"] != 0 for r in rows),
               "silent_audio_count": sum(not r["audio_has_signal"] for r in rows),
               "feature_dimensions": {"text": TEXT_DIM, "audio": AUDIO_DIM, "vision": VISION_DIM},
               "padding": "zero after valid_length, missing video frames remain zero",
               "time_alignment": "Whisper tiny.en, then base.en on low-match samples; "
                                 "fallback is energy-guided approximation",
               "whisper_aligned_count": sum(json.loads(line)["alignment_method"].startswith("whisper")
                                            for line in (PROBLEM1 / "time_map.jsonl").open(encoding="utf-8")),
               "ffmpeg": ffmpeg_version, "opencv": cv2.__version__, "numpy": np.__version__,
               "seconds": round(time.perf_counter() - started, 2)}
    (PROBLEM1 / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
