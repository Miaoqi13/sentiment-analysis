"""
BERT 数据校验脚本。
确认 train/validation/test 三个 CSV 的行数、列名、标签编码和数据质量，
确保与 BiLSTM 使用完全相同的划分。
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

EXPECTED = {
    "train.csv": 20000,
    "validation.csv": 5000,
    "test.csv": 25000,
}


def verify_file(name, expected_rows):
    path = DATA_DIR / name
    print("=" * 60)
    print(f"校验 {name}")
    print("=" * 60)

    if not path.exists():
        print(f"  [缺失] 找不到 {path}")
        return False

    df = pd.read_csv(path)
    print(f"  行数: {len(df)} (期望 {expected_rows})  ->  {'OK' if len(df) == expected_rows else '不一致'}")
    print(f"  列名: {df.columns.tolist()}")

    # 找评论列和标签列
    review_col = None
    label_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("review", "text", "content"):
            review_col = c
        if cl in ("label", "sentiment", "target", "polarity"):
            label_col = c
    print(f"  识别评论列: {review_col!r} | 标签列: {label_col!r}")

    if review_col is None or label_col is None:
        print("  [错误] 未识别到评论列或标签列")
        return False

    # 标签分布
    labels = df[label_col]
    print(f"  标签取值: {sorted(labels.dropna().unique().tolist())}")
    print(f"  标签分布:\n{labels.value_counts().sort_index().to_string()}")

    # 空值/空评论检查
    null_review = df[review_col].isna().sum()
    empty_review = (df[review_col].astype(str).str.strip() == "").sum()
    null_label = df[label_col].isna().sum()
    print(f"  空评论数: {null_review + empty_review} | 空标签数: {null_label}")

    ok = (
        len(df) == expected_rows
        and null_review == 0
        and empty_review == 0
        and null_label == 0
        and set(labels.dropna().unique().tolist()) <= {0, 1, "0", "1"}
    )
    print(f"  结论: {'通过' if ok else '存在问题'}")
    return ok


def main():
    all_ok = True
    for name, expected in EXPECTED.items():
        if not verify_file(name, expected):
            all_ok = False
    print("\n" + "=" * 60)
    print("总体校验结果:", "全部通过" if all_ok else "存在问题，请检查")
    print("=" * 60)


if __name__ == "__main__":
    main()
