
import torch

from model import BiLSTMSentimentClassifier


def main():
    vocab_size = 30000
    batch_size = 4
    seq_len = 256

    model = BiLSTMSentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3,
    )

    token_ids = torch.randint(
        low=1,
        high=vocab_size,
        size=(batch_size, seq_len),
    )

    lengths = torch.tensor([256, 200, 120, 50])

    logits = model(token_ids, lengths)

    print("输入形状：", token_ids.shape)
    print("输出形状：", logits.shape)
    print("输出分数：\n", logits)

    assert logits.shape == (batch_size, 2)
    print("\n模型自检通过！")


if __name__ == "__main__":
    main()
