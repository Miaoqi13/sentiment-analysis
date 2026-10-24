
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from dataset import SentimentDataset
from model import BiLSTMSentimentClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "outputs" / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
DROPOUT = 0.3
RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    """固定随机种子，尽量保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vocab():
    """加载前一步建立的词表。"""
    vocab_path = DATA_DIR / "vocab.json"

    if not vocab_path.exists():
        raise FileNotFoundError(
            "找不到 vocab.json，请先运行 src/vocab.py。"
        )

    with open(vocab_path, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    return vocab


def load_dataframe(filename):
    """读取已经划分好的数据。"""
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}，请先运行 src/data_utils.py。"
        )

    return pd.read_csv(path)


def calculate_metrics(labels, predictions):
    """计算二分类评价指标，以正面情感为正类。"""
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, zero_division=0
        ),
        "f1": f1_score(
            labels, predictions, zero_division=0
        ),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    """
    optimizer 不为 None 时执行训练；
    optimizer 为 None 时只评估，不更新参数。
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for token_ids, lengths, labels in loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(token_ids, lengths)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()

                # 降低循环神经网络训练中梯度爆炸的风险。
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                optimizer.step()

            total_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(logits, dim=1)
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    average_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(all_labels, all_predictions)

    return average_loss, metrics


def plot_losses(history, output_dir):
    """绘制训练集和验证集的 Loss 曲线。"""
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BiLSTM Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()


def main():
    set_seed()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("使用设备：", device)

    vocab = load_vocab()

    train_df = load_dataframe("train.csv")
    val_df = load_dataframe("validation.csv")

    train_dataset = SentimentDataset(train_df, vocab)
    val_dataset = SentimentDataset(val_df, vocab)

    # Windows 下使用 num_workers=0，减少多进程问题。
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = BiLSTMSentimentClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        dropout=DROPOUT,
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_f1 = -1.0
    history = []

    print("\n开始训练 BiLSTM...")
    print(f"训练集：{len(train_dataset)} 条")
    print(f"验证集：{len(val_dataset)} 条")
    print(f"词表大小：{len(vocab)}")
    print(f"Batch size：{BATCH_SIZE}")
    print(f"最大 Epoch：{EPOCHS}\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history.append(row)

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # 只依据验证集 F1 选择模型，不使用测试集挑选模型。
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]

            checkpoint_path = CHECKPOINT_DIR / "bilstm_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": len(vocab),
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT,
                    "max_seq_len": 256,
                    "best_val_f1": best_val_f1,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

            print(f"已保存当前最佳模型：{checkpoint_path}")

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False,
    )

    plot_losses(history_df, OUTPUT_DIR)

    print("\n训练完成。")
    print(f"最佳验证集 F1：{best_val_f1:.4f}")
    print("训练日志：", OUTPUT_DIR / "training_history.csv")
    print("Loss 曲线：", OUTPUT_DIR / "loss_curve.png")
    print("最佳模型：", CHECKPOINT_DIR / "bilstm_best.pt")


if __name__ == "__main__":
    main()

