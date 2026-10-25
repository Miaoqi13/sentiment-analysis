"""
BERT 测试集评估脚本。
- 加载 checkpoints/bert_best.pt，在 test.csv（25,000 条）上一次性评估
- 输出指标、混淆矩阵、全部预测、错例，格式与 BiLSTM 对齐
"""
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from data import BertSentimentDataset, get_tokenizer, load_dataframe

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "bert_best.pt"

BATCH_SIZE = 32
LABEL_NAMES = ["Negative", "Positive"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError("找不到 BERT 权重，请先运行 train.py。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    test_df = load_dataframe("test.csv")
    tokenizer = get_tokenizer()
    test_dataset = BertSentimentDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint["pretrained_model"], num_labels=2
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1)
            all_labels.extend(batch["labels"].tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print("\n" + "=" * 55)
    print("BERT 测试集评估结果")
    print("=" * 55)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1(正类) : {f1:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    print("\n分类报告：")
    print(classification_report(all_labels, all_preds, labels=[0, 1],
                                target_names=LABEL_NAMES, digits=4, zero_division=0))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.xlabel("Predicted Label"); plt.ylabel("True Label")
    plt.title("BERT Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bert_confusion_matrix.png", dpi=200)
    plt.close()

    # 预测结果
    result_df = test_df[["review", "label"]].copy()
    result_df["predicted_label"] = all_preds
    result_df["negative_probability"] = [p[0] for p in all_probs]
    result_df["positive_probability"] = [p[1] for p in all_probs]
    result_df["correct"] = result_df["label"] == result_df["predicted_label"]
    result_df["true_sentiment"] = result_df["label"].map({0: "negative", 1: "positive"})
    result_df["predicted_sentiment"] = result_df["predicted_label"].map({0: "negative", 1: "positive"})
    result_df.to_csv(OUTPUT_DIR / "bert_test_predictions.csv", index=False)

    errors_df = result_df[~result_df["correct"]].copy()
    errors_df.to_csv(OUTPUT_DIR / "bert_misclassified_reviews.csv", index=False)

    metrics = {
        "model": "BERT",
        "pretrained": checkpoint["pretrained_model"],
        "test_size": len(test_df),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_validation_f1": checkpoint.get("best_val_f1"),
        "errors": len(errors_df),
    }
    with open(OUTPUT_DIR / "bert_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n结果已保存到：", OUTPUT_DIR)
    print("混淆矩阵：bert_confusion_matrix.png")
    print("全部预测：bert_test_predictions.csv")
    print("错误样本：bert_misclassified_reviews.csv")
    print("指标文件：bert_metrics.json")
    print("测试集错误数量：", len(errors_df))


if __name__ == "__main__":
    main()
