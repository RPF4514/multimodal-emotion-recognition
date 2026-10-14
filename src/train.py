from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .data import deterministic_mask, load_datasets, mask_batch, reencode_masked_text, to_device
from .metrics import all_metrics
from .models import AsyncLateEmotionModel, EmotionModel, TextStudent
from .paths import CHECKPOINTS, LOGS, ensure_dirs
from .text_bert_encoder import load_bert

PATTERNS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_student(path: Path | None = None):
    path = path or CHECKPOINTS / "text_student.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run python -m src.distill_text first")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = saved.get("student_config", {})
    student = TextStudent(dim=config.get("dim", 128),
                          num_layers=config.get("num_layers", 1))
    student.load_state_dict(saved["state_dict"])
    return student


def make_model(kind: str, hidden: int, device, text_source="student",
               fine_tune_text_student=False, student_checkpoint: Path | None = None,
               bridge_dropout=0.25, class_mode="softmax"):
    if kind == "async_late_bigru":
        if text_source != "student" or fine_tune_text_student or class_mode != "softmax":
            raise ValueError("The async model requires a frozen student encoder and softmax class head.")
        return AsyncLateEmotionModel(load_student(student_checkpoint), hidden=hidden).to(device)
    return EmotionModel(kind, load_student(student_checkpoint), hidden=hidden, text_source=text_source,
                        train_text_student=fine_tune_text_student,
                        bridge_dropout=bridge_dropout, class_mode=class_mode).to(device)


def evaluate_loader(model, loader, device, corrupted=False, bert_encoder=None):
    model.eval()
    truth_cls, truth_reg, pred_cls, pred_reg = [], [], [], []
    elapsed = 0.0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if corrupted:
                batch = deterministic_mask(batch, PATTERNS[batch_index % len(PATTERNS)], 0.3)
            batch = to_device(batch, device)
            if (corrupted and bert_encoder is not None and "teacher_text" in batch
                    and bool((~batch["observed"][..., 0] & batch["content"]).any())):
                batch = reencode_masked_text(batch, bert_encoder)
            start = time.perf_counter()
            output = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            truth_cls.extend(batch["cls"].cpu().tolist())
            truth_reg.extend(batch["reg"].cpu().tolist())
            pred_cls.extend(output["logits"].argmax(-1).cpu().tolist())
            pred_reg.extend(output["strength"].cpu().tolist())
    metrics = all_metrics(truth_cls, pred_cls, truth_reg, pred_reg)
    metrics["inference_seconds"] = elapsed
    return metrics


def train_epoch(model, loader, optimizer, device, class_weights, rng, regression_weight=0.4,
                consistency_weight=0.1, missing_augmentation=True, adapter_weight=0.3,
                bert_encoder=None):
    model.train()
    total = 0.0
    for batch in loader:
        masked = mask_batch(batch, rng) if missing_augmentation else batch
        batch = to_device(batch, device)
        masked = to_device(masked, device)
        if missing_augmentation and bert_encoder is not None and "teacher_text" in masked:
            # A/V-only masks need no re-encoding; text masks change all BERT
            # contextual positions, including the surviving special tokens.
            text_changed = (batch["text_mask"] != masked["text_mask"]).any()
            if bool(text_changed):
                masked = reencode_masked_text(masked, bert_encoder)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            clean = model(batch)
            damaged = model(masked)
            ce = (F.cross_entropy(clean["logits"], batch["cls"], weight=class_weights) +
                  F.cross_entropy(damaged["logits"], batch["cls"], weight=class_weights)) / 2
            reg = (F.smooth_l1_loss(clean["strength"], batch["reg"]) +
                   F.smooth_l1_loss(damaged["strength"], batch["reg"])) / 2
            consistent = F.kl_div(F.log_softmax(damaged["logits"], dim=-1),
                                  F.softmax(clean["logits"].detach(), dim=-1),
                                  reduction="batchmean")
            consistent += F.smooth_l1_loss(damaged["strength"], clean["strength"].detach())
            loss = ce + regression_weight * reg + consistency_weight * consistent
            adapter_terms = [out["recon"] for out in (clean, damaged)
                             if out.get("recon") is not None]
            if adapter_terms:
                # Trains the 768-d text bridge that serves feature-poor attachments.
                loss = loss + adapter_weight * torch.stack(adapter_terms).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach()) * len(batch["cls"])
    return total / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline_bigru", "gated_bigru", "gated_transformer",
                                            "async_late_bigru"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--text-lr", type=float, default=1e-5,
                        help="Learning rate for the student text encoder when fine-tuning it")
    parser.add_argument("--text-source", choices=("student", "teacher", "bridge"), default="student")
    parser.add_argument("--class-mode", choices=("softmax", "ordinal"), default="softmax")
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--neutral-weight-multiplier", type=float, default=1.0,
                        help="Multiply only the Neutral cross-entropy class weight")
    parser.add_argument("--regression-weight", type=float, default=0.4)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--bridge-dropout", type=float, default=0.25)
    parser.add_argument("--adapter-weight", type=float, default=0.3)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--student-checkpoint", type=Path,
                        default=CHECKPOINTS / "text_student.pt")
    parser.add_argument("--fine-tune-text-student", action="store_true")
    parser.add_argument("--no-missing-augmentation", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0, help="For smoke checks only")
    parser.add_argument("--mask-protocol", choices=("auto", "content_only_20260924",
                                                     "bert_reencoded_20260924"), default="auto")
    parser.add_argument("--bert-dir", type=Path,
                        default=LOGS.parent / "bert" / "bert-base-uncased")
    args = parser.parse_args()
    ensure_dirs()
    seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_protocol = ("bert_reencoded_20260924" if args.text_source in ("teacher", "bridge")
                     else "content_only_20260924") if args.mask_protocol == "auto" else args.mask_protocol
    if args.text_source in ("teacher", "bridge") and mask_protocol == "content_only_20260924":
        parser.error("Contextual text masking requires bert_reencoded_20260924")
    if args.text_source == "student" and mask_protocol == "bert_reencoded_20260924":
        parser.error("Student text has no precomputed BERT features to re-encode")
    bert_encoder = (load_bert(args.bert_dir, device) if mask_protocol == "bert_reencoded_20260924"
                    and args.text_source in ("teacher", "bridge") else None)
    datasets, _ = load_datasets()
    train_set = datasets["train"]
    if args.limit_train:
        train_set = Subset(train_set, list(range(min(args.limit_train, len(train_set)))))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(datasets["valid"], batch_size=args.batch_size * 2,
                              shuffle=False, pin_memory=True)
    if args.text_source == "teacher" and args.fine_tune_text_student:
        parser.error("--fine-tune-text-student requires --text-source student")
    model = make_model(args.model, args.hidden, device, args.text_source,
                       args.fine_tune_text_student, args.student_checkpoint, args.bridge_dropout,
                       args.class_mode)
    if args.fine_tune_text_student:
        text_params = [p for p in model.text_student.parameters() if p.requires_grad]
        text_param_ids = {id(p) for p in text_params}
        fusion_params = [p for p in model.parameters()
                         if p.requires_grad and id(p) not in text_param_ids]
        optimizer = torch.optim.AdamW([
            {"params": fusion_params, "lr": args.lr},
            {"params": text_params, "lr": args.text_lr},
        ], weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=args.lr, weight_decay=0.01)
    class_counts = torch.bincount(datasets["train"].cls, minlength=3).float()
    class_weights = (class_counts.sum() / (3 * class_counts)).pow(args.class_weight_power).to(device)
    if args.neutral_weight_multiplier <= 0:
        parser.error("--neutral-weight-multiplier must be positive")
    class_weights[1] *= args.neutral_weight_multiplier
    rng = np.random.default_rng(args.seed)
    best_score, bad_epochs, history = -1.0, 0, []
    name = f"{args.model}{'_' + args.run_tag if args.run_tag else ''}_seed{args.seed}"
    output = CHECKPOINTS / f"{name}.pt"
    if output.exists():
        parser.error(f"Refusing to overwrite an existing checkpoint: {output}")
    for number in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, device, class_weights, rng,
                                 args.regression_weight, args.consistency_weight,
                                 not args.no_missing_augmentation,
                                 adapter_weight=args.adapter_weight, bert_encoder=bert_encoder)
        clean = evaluate_loader(model, valid_loader, device, corrupted=False)
        damaged = evaluate_loader(model, valid_loader, device, corrupted=True,
                                  bert_encoder=bert_encoder)
        score = (clean["macro_f1"] + damaged["macro_f1"]) / 2
        row = {"epoch": number, "train_loss": train_loss,
               "clean_macro_f1": clean["macro_f1"], "damaged_macro_f1": damaged["macro_f1"],
               "clean_mae": clean["mae"], "damaged_mae": damaged["mae"],
               "selection_score": score, "seconds": round(time.perf_counter() - start, 2)}
        history.append(row)
        print(json.dumps({"model": args.model, "seed": args.seed, **row}), flush=True)
        if score > best_score + 1e-5:
            best_score, bad_epochs = score, 0
            torch.save({"model": args.model, "hidden": args.hidden, "seed": args.seed,
                        "text_source": args.text_source, "run_tag": args.run_tag,
                        "class_mode": args.class_mode,
                        "neutral_weight_multiplier": args.neutral_weight_multiplier,
                        "student_checkpoint": str(args.student_checkpoint),
                        "student_config": {"dim": model.text_student.dim,
                                           "num_layers": model.text_student.num_layers},
                        "mask_protocol": mask_protocol,
                        "train_text_student": args.fine_tune_text_student,
                        "state_dict": model.state_dict(), "epoch": number,
                        "selection_score": score, "valid_clean": clean,
                        "valid_damaged": damaged}, output)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    log_config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}
    log = {"model": args.model, "seed": args.seed, "config": log_config,
           "best_score": best_score, "checkpoint": str(output), "history": history,
           "mask_protocol": mask_protocol,
           "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "total_params": sum(p.numel() for p in model.parameters()),
           "checkpoint_bytes": output.stat().st_size,
           "gpu_peak_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
           "smoke_only": bool(args.limit_train)}
    suffix = "_smoke" if args.limit_train else ""
    (LOGS / f"{name}{suffix}.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: log[key] for key in ("model", "seed", "best_score", "checkpoint_bytes",
                                                 "trainable_params", "gpu_peak_bytes")}), flush=True)


if __name__ == "__main__":
    main()
