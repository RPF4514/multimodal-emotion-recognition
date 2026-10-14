"""Fetch the public frozen BERT dependency and reject unverified bytes.

This is a deployment dependency, not additional sentiment-training data.
The E-problem expert forum explicitly exempts public general pretrained
feature-extractor weights from the <=50M attachment; the version, official
download source and verification procedure must remain disclosed.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

from .paths import OUT


FILES = {
    "pytorch_model.bin": "097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a",
    "config.json": "7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20",
}
BASE = "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=OUT / "bert" / "bert-base-uncased")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = args.dest / name
        if target.exists():
            actual = sha256(target)
            if actual != expected:
                raise RuntimeError(f"Existing {target} has wrong SHA-256: {actual}")
            print(f"VERIFIED {target} {actual}")
            continue
        if args.verify_only:
            raise FileNotFoundError(f"Required BERT file absent: {target}")
        partial = target.with_name(target.name + ".partial")
        if partial.exists():
            raise FileExistsError(f"Remove or inspect incomplete download first: {partial}")
        try:
            with urlopen(BASE + name, timeout=60) as response, partial.open("xb") as output:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
            actual = sha256(partial)
            if actual != expected:
                raise RuntimeError(f"Downloaded {name} has wrong SHA-256: {actual}")
            partial.replace(target)
            print(f"DOWNLOADED AND VERIFIED {target} {actual}")
        except Exception:
            # Keep the partial file for diagnosis; never install unverified bytes.
            raise


if __name__ == "__main__":
    main()
