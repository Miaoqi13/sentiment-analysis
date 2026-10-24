
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from dataset import SentimentDataset
from model import BiLSTMSentimentClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "outputs" / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "bilstm_best.pt"

BATCH_SIZE = 64
LABEL_NAMES = ["Negative", "Positive"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "找不到最佳模型权重，请先运行 train.py。"
        )

    test_path = DATA_DIR / "test.csv"
    vocab_path = DATA_DIR / "vocab.json"

    if not test_path.exists() or not vocab_path.exists():
        raise FileNotFoundError(
            "缺少 test.csv 或 vocab.json，请先完成数据准备。"
        )

    test_df = pd.read_csv(test_path)

    with open(vocab_path, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("使用设备：", device)

    # weights_only=False 适用于本脚本保存的 checkpoint。
    # 只加载自己训练并信任的模型文件。
    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # 兼容不支持 weights_only 参数的旧版 PyTorch。
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )

    model = BiLSTMSentimentClassifier(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=1,
        dropout=checkpoint["dropout"],
        num_classes=2,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = SentimentDataset(test_df, vocab)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for token_ids, lengths, labels in test_loader:
            token_ids = token_ids.to(device)

            logits = model(token_ids, lengths)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(
                predictions.cpu().tolist()
            )
            all_probabilities.extend(
                probabilities.cpu().tolist()
            )

    accuracy = accuracy_score(
        all_labels, all_predictions
    )
    precision = precision_score(
        all_labels, all_predictions, zero_division=0
    )
    recall = recall_score(
        all_labels, all_predictions, zero_division=0
    )
    f1 = f1_score(
        all_labels, all_predictions, zero_division=0
    )

    print("\n" + "=" * 55)
    print("BiLSTM 测试集评估结果")
    print("=" * 55)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\n分类报告：")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=LABEL_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    # 保存混淆矩阵。
    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1],
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("BiLSTM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=200,
    )
    plt.close()

    # 保存每条评论的预测结果。
    result_df = test_df[["review", "label"]].copy()
    result_df["predicted_label"] = all_predictions
    result_df["negative_probability"] = [
        prob[0] for prob in all_probabilities
    ]
    result_df["positive_probability"] = [
        prob[1] for prob in all_probabilities
    ]
    result_df["correct"] = (
        result_df["label"] == result_df["predicted_label"]
    )

    result_df["true_sentiment"] = result_df["label"].map(
        {0: "negative", 1: "positive"}
    )
    result_df["predicted_sentiment"] = (
        result_df["predicted_label"].map(
            {0: "negative", 1: "positive"}
        )
    )

    result_df.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    # 导出错误样本，便于定性分析。
    errors_df = result_df[
        ~result_df["correct"]
    ].copy()

    errors_df.to_csv(
        OUTPUT_DIR / "misclassified_reviews.csv",
        index=False,
    )

    metrics = {
        "model": "BiLSTM",
        "test_size": len(test_df),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_validation_f1": checkpoint.get("best_val_f1"),
    }

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print("\n结果已保存到：", OUTPUT_DIR)
    print("混淆矩阵：confusion_matrix.png")
    print("全部预测：test_predictions.csv")
    print("错误样本：misclassified_reviews.csv")
    print("指标文件：metrics.json")
    print("测试集错误数量：", len(errors_df))


if __name__ == "__main__":
    main()
