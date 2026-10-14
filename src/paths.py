import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ["MM_EMOTION_DATA_ROOT"]).resolve() if os.environ.get(
    "MM_EMOTION_DATA_ROOT") else PROJECT.parent / "E题数据" / "E题数据"
ALIGNED = DATA_ROOT / "附件2-数据集特征文件" / "aligned_50.pkl"
LABELS = DATA_ROOT / "附件2-数据集特征文件" / "label.xlsx"
ATT3 = DATA_ROOT / "附件3-模态缺失特征样本" / "对齐版本"
ATT4 = DATA_ROOT / "附件4-可解释专项视频样本与特征文件" / "附件4-可解释专项视频样本与特征文件" / "对齐版本"
ATT1 = DATA_ROOT / "附件1-数据集原始多模态样本" / "MOSEI数据集部分原始视频-100条"
OUT = PROJECT / "outputs"
CHECKPOINTS = OUT / "checkpoints"
LOGS = OUT / "logs"
FIGURES = OUT / "figures"
PREDICTIONS = OUT / "predictions"
PROBLEM1 = OUT / "problem1"


def ensure_dirs() -> None:
    for path in (CHECKPOINTS, LOGS, FIGURES, PREDICTIONS, PROBLEM1):
        path.mkdir(parents=True, exist_ok=True)


def selected_checkpoint_path(decision: dict) -> Path:
    stored = Path(decision["selected_checkpoint"])
    # Selection metadata may retain the absolute path from the training machine.
    # A copied submission must resolve its own checkpoint first, never silently
    # reach back into that machine's project during an independent reproduction.
    local = CHECKPOINTS / stored.name
    if local.is_file():
        return local
    if stored.is_absolute() and not stored.is_relative_to(PROJECT):
        raise FileNotFoundError(
            f"Local checkpoint missing: {local}; refusing external training path {stored}")
    if stored.is_file():
        return stored
    raise FileNotFoundError(f"Selected checkpoint not found: {local} or {stored}")
