# BERT 情感分类（组员：刘嘉慧）

基于 `bert-base-uncased` 的 IMDb 电影评论二分类（负面=0，正面=1）。
与 BiLSTM 使用**完全相同**的 train/validation/test 划分和评价指标口径。

## 目录

```
src/bert/
├── verify_data.py     # 校验三个 CSV 的行数/列名/标签
├── data.py            # BERT tokenizer 编码与 Dataset
├── train.py           # 训练，按验证集正面类 F1 保存最佳权重
└── evaluate.py        # 测试集评估，输出指标/混淆矩阵/预测/错例
```

## 环境

```bash
pip install -r requirements.txt
# GPU 版 torch（RTX 4060 / CUDA 12.x）：
# pip install torch --index-url https://download.pytorch.org/whl/cu124
# 国内下载模型可加环境变量：
# set HF_ENDPOINT=https://hf-mirror.com
```

## 数据

把组员切好的三个 CSV 放到 `data/processed/`：

```
data/processed/
├── train.csv        # 20,000 条
├── validation.csv  #  5,000 条
└── test.csv         # 25,000 条
```

要求列名为 `review,label`，标签为整数 0/1。运行 `python src/bert/verify_data.py` 校验。

## 运行（在 src/bert 目录下）

```bash
cd src/bert

# 1. 校验数据
python verify_data.py

# 2. 训练（首次会自动下载 bert-base-uncased）
python train.py
#   产物：checkpoints/bert_best.pt
#         outputs/training/bert_training_history.csv
#         outputs/training/bert_loss_curve.png

# 3. 测试集评估（复用最佳权重）
python evaluate.py
#   产物：outputs/evaluation/bert_metrics.json
#         outputs/evaluation/bert_confusion_matrix.png
#         outputs/evaluation/bert_test_predictions.csv
#         outputs/evaluation/bert_misclassified_reviews.csv
```

## 超参数

| 参数 | 值 |
|---|---|
| 预训练模型 | bert-base-uncased |
| max_length | 256（与 BiLSTM 一致） |
| batch_size | 16（8GB 显存） |
| epochs | 3 |
| optimizer | AdamW, lr=2e-5, weight_decay=0.01 |
| warmup | 10% |
| 模型选择依据 | 验证集正面类 F1（与 BiLSTM 同口径） |

## 结果（test 25,000 条）

| 指标 | BiLSTM | BERT |
|---|---|---|
| Accuracy | 0.8286 | 0.9219 |
| Precision（正面） | 0.8041 | 0.9173 |
| Recall（正面） | 0.8688 | 0.9274 |
| F1（正面） | 0.8352 | 0.9223 |
| Macro F1 | 0.8283 | 0.9219 |

BERT 训练约 31 分钟（RTX 4060 Laptop），109.5M 参数。
