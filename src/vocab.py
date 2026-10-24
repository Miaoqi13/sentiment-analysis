
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "outputs" / "data"

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MAX_VOCAB_SIZE = 30000
MAX_SEQ_LEN = 256

# 保留单词内部的撇号，例如 don't、isn't。
TOKEN_PATTERN = re.compile(
    r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]",
    flags=re.IGNORECASE,
)


def tokenize(text):
    """将一条已经清洗过的评论切分成 token。"""
    return TOKEN_PATTERN.findall(str(text).lower())


def build_vocab(train_df, max_vocab_size=MAX_VOCAB_SIZE):
    """只根据训练集构建词表。"""
    counter = Counter()

    for review in train_df["review"]:
        counter.update(tokenize(review))

    # 预留两个位置给 PAD 和 UNK。
    most_common = counter.most_common(max_vocab_size - 2)

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for token, _ in most_common:
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def encode_text(text, vocab, max_seq_len=MAX_SEQ_LEN):
    """把文本编码成定长整数序列。"""
    tokens = tokenize(text)

    ids = [
        vocab.get(token, vocab[UNK_TOKEN])
        for token in tokens[:max_seq_len]
    ]

    # 不足 max_seq_len 时用 PAD 补齐。
    if len(ids) < max_seq_len:
        ids.extend([vocab[PAD_TOKEN]] * (max_seq_len - len(ids)))

    return ids


def main():
    train_path = DATA_DIR / "train.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"找不到 {train_path}。\n"
            "请先运行 data_utils.py 完成数据划分。"
        )

    train_df = pd.read_csv(train_path)

    vocab = build_vocab(train_df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    vocab_path = DATA_DIR / "vocab.json"

    with open(vocab_path, "w", encoding="utf-8") as file:
        json.dump(vocab, file, ensure_ascii=False)

    print("词表构建完成。")
    print("训练集评论数：", len(train_df))
    print("词表大小：", len(vocab))
    print("PAD 编号：", vocab[PAD_TOKEN])
    print("UNK 编号：", vocab[UNK_TOKEN])

    example = train_df.iloc[0]["review"]
    encoded = encode_text(example, vocab)

    print("\n示例评论：")
    print(example[:200])
    print("\n前 20 个 token：")
    print(tokenize(example)[:20])
    print("\n编码序列前 20 项：")
    print(encoded[:20])
    print("编码序列长度：", len(encoded))
    print("\n词表保存位置：", vocab_path)


if __name__ == "__main__":
    main()
