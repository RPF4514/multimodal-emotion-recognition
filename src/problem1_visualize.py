"""Make a traceable example from the generated Problem 1 time map."""
from __future__ import annotations

import json
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paths import FIGURES, PROBLEM1
from .problem1 import SAMPLE_RATE, decode_audio, find_ffmpeg


def main():
    source = PROBLEM1 / "time_map.jsonl"
    record = json.loads(source.open(encoding="utf-8").readline())
    ffmpeg = find_ffmpeg()
    audio, _, _ = decode_audio(record["file"], ffmpeg)
    positions = [p for p in record["positions"] if p["video_frame_decoded"]]
    chosen = positions[2:6]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), gridspec_kw={"height_ratios": [1, 2]})
    axis = axes[0]
    sample = max(1, len(audio) // 5000)
    axis.plot(np.arange(0, len(audio), sample) / SAMPLE_RATE, audio[::sample], lw=0.35)
    for row in chosen:
        axis.axvspan(row["start_second"], row["end_second"], alpha=0.2)
    axis.set(xlabel="Seconds from original video", ylabel="Audio amplitude",
             title=f"ASR-anchored word alignment for {record['video_id']} clip {record['clip_id']}")
    axis.grid(alpha=0.2)
    axes[1].axis("off")
    for j, row in enumerate(chosen):
        second = (row["start_second"] + row["end_second"]) / 2
        decoded = subprocess.run([str(ffmpeg), "-v", "error", "-ss", str(second),
                                  "-i", record["file"], "-frames:v", "1", "-vf",
                                  "scale=320:180", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
                                 capture_output=True, timeout=30)
        frame = cv2.imdecode(np.frombuffer(decoded.stdout, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        inset = axes[1].inset_axes([j * 0.25 + 0.01, 0.18, 0.23, 0.72])
        inset.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inset.axis("off")
        inset.set_title(f"step {row['step']}  {second:.2f}s", fontsize=8)
        axes[1].text(j * 0.25 + 0.125, 0.08, row["text"], ha="center", va="center",
                     fontsize=9, transform=axes[1].transAxes)
    fig.tight_layout()
    output = FIGURES / "problem1_typical_alignment.png"
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
