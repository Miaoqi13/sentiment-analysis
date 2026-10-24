
import html
import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split


# 项目根目录：src 文件夹的上一级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "IMDb Dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "data"

RANDOM_SEED = 42
MAX_VOCAB_SIZE = 30000
MAX_SEQ_LEN = 256


def clean_text(text):
    """清洗评论文本，保留否定词和基本标点。"""
    if pd.isna(text):
        return ""

    text = html.unescape(str(text))
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_label(label):
    """将常见情感标签统一为 0（负面）或 1（正面）。"""
    if pd.isna(label):
        raise ValueError("发现空的情感标签。")

    value = str(label).strip().lower()

    positive_labels = {"positive", "pos", "1", "1.0"}
    negative_labels = {"negative", "neg", "0", "0.0"}

    if value in positive_labels:
        return 1

    if value in negative_labels:
        return 0

    raise ValueError(
        f"无法识别情感标签：{label!r}。"
        "请检查 CSV 中的标签列。"
    )


def find_column(columns, candidates):
    """根据候选名称查找列，忽略大小写和首尾空格。"""
    normalized = {
        str(column).strip().lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    return None


def load_dataset():
    """读取并检查 IMDb CSV。"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{DATA_PATH}\n"
            "请确认 CSV 已放入 data/raw/ 目录。"
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("原始数据检查")
    print("=" * 60)
    print("文件路径：", DATA_PATH)
    print("数据形状：", df.shape)
    print("列名：", df.columns.tolist())
    print("\n前 5 行：")
    print(df.head())
    print("\n各列缺失值数量：")
    print(df.isna().sum())

    review_col = find_column(
        df.columns,
        ["review", "text", "content", "comment"]
    )
    label_col = find_column(
        df.columns,
        ["sentiment", "label", "target", "polarity"]
    )
    split_col = find_column(
        df.columns,
        ["split", "dataset", "set", "subset"]
    )

    if review_col is None or label_col is None:
        raise ValueError(
            "没有找到评论列或情感标签列。\n"
            f"当前列名：{df.columns.tolist()}\n"
            "请确认评论列类似 review，标签列类似 sentiment。"
        )

    data = pd.DataFrame({
        "review": df[review_col].map(clean_text),
        "label": df[label_col].map(normalize_label),
    })

    # 只有原始数据明确包含划分列时，才使用该列识别官方划分。
    if split_col is not None:
        data["original_split"] = (
            df[split_col].astype("string").str.strip().str.lower()
        )
    else:
        data["original_split"] = "unspecified"

    # 删除空评论，重复评论暂不删除，避免擅自改变实验数据。
    before = len(data)
    data = data[data["review"].str.len() > 0].copy()
    removed = before - len(data)

    data = data.reset_index(drop=True)

    print("\n清洗后数据量：", len(data))
    print("删除的空评论数：", removed)
    print("情感标签分布：")
    print(data["label"].value_counts().sort_index())

    if data["label"].nunique() != 2:
        raise ValueError("当前数据不是二分类数据，请检查标签。")

    return data


def split_dataset(data):
    """
    若存在明确的 train/test 划分列，则保留测试集；
    否则对整份数据做分层随机划分。

    对常见的 50,000 条合并版 IMDb CSV：
    train=20,000，validation=5,000，test=25,000。
    注意：这种情况下 test 是随机留出集，不是官方测试集。
    """
    split_values = set(data["original_split"].dropna().unique())

    train_names = {"train", "training"}
    test_names = {"test", "testing"}

    has_official_style_split = (
        split_values.intersection(train_names)
        and split_values.intersection(test_names)
    )

    if has_official_style_split:
        original_train = data[
            data["original_split"].isin(train_names)
        ].copy()

        test_df = data[
            data["original_split"].isin(test_names)
        ].copy()

        if len(original_train) < 2 or len(test_df) == 0:
            raise ValueError("原始 train/test 划分数量不足。")

        train_df, val_df = train_test_split(
            original_train,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=original_train["label"],
        )

        print("\n检测到原始 train/test 划分列。")
        print("保留原始测试集，验证集从原始训练集划分。")

    else:
        if len(data) != 50000:
            print(
                "\n提示：当前数据不是 50,000 条，"
                "将按 80%/20% 两阶段划分，"
                "实际数量取决于数据集大小。"
            )

        # 第一步：25% 留作测试集。
        train_val_df, test_df = train_test_split(
            data,
            test_size=0.5 if len(data) == 50000 else 0.2,
            random_state=RANDOM_SEED,
            stratify=data["label"],
        )

        # 对 50,000 条数据，剩余 25,000 条再划分为
        # 20,000 条训练集和 5,000 条验证集。
        if len(data) == 50000:
            val_ratio = 5000 / 25000
        else:
            val_ratio = 0.2

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=RANDOM_SEED,
            stratify=train_val_df["label"],
        )

        print("\n未检测到原始 train/test 划分列。")
        print("本次采用分层随机划分，测试集不是官方测试集。")

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df):
    """保存清洗后的划分结果，便于复现实验。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df[["review", "label"]].to_csv(
        OUTPUT_DIR / "train.csv", index=False
    )
    val_df[["review", "label"]].to_csv(
        OUTPUT_DIR / "validation.csv", index=False
    )
    test_df[["review", "label"]].to_csv(
        OUTPUT_DIR / "test.csv", index=False
    )

    metadata = {
        "random_seed": RANDOM_SEED,
        "max_vocab_size": MAX_VOCAB_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "train_size": len(train_df),
        "validation_size": len(val_df),
        "test_size": len(test_df),
        "label_mapping": {
            "negative": 0,
            "positive": 1,
        },
    }

    with open(
        OUTPUT_DIR / "split_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("\n数据划分结果：")
    print(f"Train      : {len(train_df)}")
    print(f"Validation : {len(val_df)}")
    print(f"Test       : {len(test_df)}")
    print("\n训练集标签分布：")
    print(train_df["label"].value_counts().sort_index())
    print("\n验证集标签分布：")
    print(val_df["label"].value_counts().sort_index())
    print("\n测试集标签分布：")
    print(test_df["label"].value_counts().sort_index())
    print("\n划分文件已保存到：", OUTPUT_DIR)


def main():
    data = load_dataset()
    train_df, val_df, test_df = split_dataset(data)
    save_splits(train_df, val_df, test_df)

    print("\n数据准备完成。")


if __name__ == "__main__":
    main()
