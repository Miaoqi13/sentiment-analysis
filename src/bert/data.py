"""
BERT 数据处理模块。
- 读取与 BiLSTM 相同的三个 CSV（data/processed/）
- 使用 bert-base-uncased 自带 tokenizer 编码（不使用 BiLSTM 的 vocab）
- 标签统一为整数：负面=0，正面=1
"""
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

PRETRAINED_MODEL = "bert-base-uncased"
MAX_LENGTH = 256  # 与 BiLSTM 的 MAX_SEQ_LEN=256 对齐


def load_dataframe(filename):
    """读取已划分好的 CSV（review, label）。"""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    df = pd.read_csv(path)

    # 兼容列名差异：评论列可能叫 review/text，标签列可能叫 label/sentiment
    review_col = next(c for c in df.columns if str(c).strip().lower() in ("review", "text", "content"))
    label_col = next(c for c in df.columns if str(c).strip().lower() in ("label", "sentiment", "target", "polarity"))

    out = pd.DataFrame({
        "review": df[review_col].astype(str),
        "label": df[label_col].astype(int),  # CSV 中已是 0/1
    })
    return out


class BertSentimentDataset(Dataset):
    """对评论文本做 BERT tokenizer 编码。"""

    def __init__(self, df, tokenizer, max_length=MAX_LENGTH):
        self.texts = df["review"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_tokenizer():
    return AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
