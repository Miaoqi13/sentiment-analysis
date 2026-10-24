"""
BERT 训练脚本。
- 模型：bert-base-uncased 二分类（负面=0，正面=1）
- 数据：与 BiLSTM 完全相同的 train/validation 划分
- 模型选择：以验证集正面类 F1（与 BiLSTM 同口径）保存最佳权重
- 输出：训练历史 CSV、Loss 曲线、最佳权重
"""
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from data import BertSentimentDataset, get_tokenizer, load_dataframe, PRETRAINED_MODEL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# 超参数
BATCH_SIZE = 16          # RTX 4060 8GB：bert-base + seq256，batch=16 安全
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
PATIENCE = 2             # 验证集 F1 连续 2 轮不提升则早停
RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(labels, preds):
    """与 BiLSTM 同口径：以正面类(1)为正类。"""
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),          # 正面类 F1（模型选择依据）
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    all_labels, all_preds = [], []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, compute_metrics(all_labels, all_preds)


def plot_losses(history, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BERT Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "bert_loss_curve.png", dpi=200)
    plt.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)
    if device.type == "cuda":
        print("GPU：", torch.cuda.get_device_name(0))

    tokenizer = get_tokenizer()
    train_df = load_dataframe("train.csv")
    val_df = load_dataframe("validation.csv")

    train_dataset = BertSentimentDataset(train_df, tokenizer)
    val_dataset = BertSentimentDataset(val_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("加载预训练模型：", PRETRAINED_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL, num_labels=2
    ).to(device)
    print("可训练参数规模：", f"{count_parameters(model)/1e6:.1f}M")

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped, lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    history = []
    epochs_no_improve = 0
    total_train_start = time.time()

    print("\n开始训练 BERT...")
    print(f"训练集：{len(train_dataset)} 条 | 验证集：{len(val_dataset)} 条")
    print(f"Batch size：{BATCH_SIZE} | 最大 Epoch：{EPOCHS} | LR：{LEARNING_RATE}\n")

    for epoch in range(1, EPOCHS + 1):
        ep_start = time.time()
        train_loss, train_metrics = run_epoch(model, train_loader, device, optimizer, scheduler)
        val_loss, val_metrics = run_epoch(model, val_loader, device)
        ep_time = time.time() - ep_start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": round(ep_time, 1),
        }
        history.append(row)

        print(
            f"Epoch [{epoch}/{EPOCHS}] {ep_time:.0f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val P: {val_metrics['precision']:.4f} | "
            f"Val R: {val_metrics['recall']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val MacroF1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "pretrained_model": PRETRAINED_MODEL,
                    "epoch": epoch,
                    "best_val_f1": best_val_f1,
                    "best_val_macro_f1": val_metrics["macro_f1"],
                    "max_length": 256,
                },
                CHECKPOINT_DIR / "bert_best.pt",
            )
            print(f"  -> 保存最佳模型 (Val F1={best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  -> Val F1 未提升 ({epochs_no_improve}/{PATIENCE})")
            if epochs_no_improve >= PATIENCE:
                print("触发早停。")
                break

    total_train_time = time.time() - total_train_start

    import pandas as pd
    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "bert_training_history.csv", index=False)
    plot_losses(history_df, OUTPUT_DIR)

    summary = {
        "model": "BERT",
        "pretrained": PRETRAINED_MODEL,
        "best_val_f1": best_val_f1,
        "best_val_macro_f1": max(r["val_macro_f1"] for r in history),
        "total_train_seconds": round(total_train_time, 1),
        "parameters_M": round(count_parameters(model) / 1e6, 1),
        "epochs_run": len(history),
    }
    with open(OUTPUT_DIR / "bert_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n训练完成。")
    print(f"总训练时间：{total_train_time:.0f}s")
    print(f"最佳验证集 F1：{best_val_f1:.4f}")
    print("历史：", OUTPUT_DIR / "bert_training_history.csv")
    print("权重：", CHECKPOINT_DIR / "bert_best.pt")


if __name__ == "__main__":
    main()
