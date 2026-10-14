"""Independent attachment-4 check of supplied and frozen-BERT text vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_pickle
from .paths import ATT4, OUT, PROBLEM1
from .text_bert_encoder import encode, load_bert


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-dir", type=Path, default=OUT / "bert" / "bert-base-uncased")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = load_bert(args.bert_dir, "cuda" if torch.cuda.is_available() else "cpu")
    mapping = {
        row["file"]: row
        for row in (
            json.loads(line)
            for line in (PROBLEM1 / "attachment4_time_map.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        )
    }
    rows = []
    for path in sorted(ATT4.glob("*.pkl")):
        raw = load_pickle(path)
        tokens = np.asarray(raw["text_bert"])
        reference = np.asarray(raw["text"], dtype=np.float32)
        if tokens.ndim == 2:
            tokens = tokens[None]
        if reference.ndim == 2:
            reference = reference[None]
        actual = encode(model, torch.from_numpy(tokens).long()).numpy()
        if actual.shape != reference.shape:
            raise ValueError(f"Shape mismatch in {path.name}: {actual.shape} != {reference.shape}")
        mask = tokens[:, 1, :].astype(bool)
        actual_valid = actual[mask].reshape(-1)
        reference_valid = reference[mask].reshape(-1)
        cosine = float(np.dot(actual_valid, reference_valid) /
                       (np.linalg.norm(actual_valid) * np.linalg.norm(reference_valid)))
        max_abs = float(np.max(np.abs(actual - reference)))
        rows.append({"file": path.name, "cosine": cosine,
                     "max_abs_diff": max_abs,
                     "raw_text_token_prefix_match": bool(mapping[path.name]["bert_prefix_matches"]),
                     "consistent": bool(cosine >= 0.99999 and max_abs <= 0.001 and
                                        mapping[path.name]["bert_prefix_matches"])})
    result = {"checked": len(rows), "consistent": sum(row["consistent"] for row in rows),
              "min_cosine": min(row["cosine"] for row in rows),
              "max_abs_diff": max(row["max_abs_diff"] for row in rows),
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"},
                     ensure_ascii=False))
    if result["checked"] != 20 or result["consistent"] != 20:
        raise SystemExit("Attachment-4 text consistency audit failed")


if __name__ == "__main__":
    main()
