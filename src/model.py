
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTMSentimentClassifier(nn.Module):
    """基于双向 LSTM 的二分类情感分析模型。"""

    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3,
        num_classes=2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, token_ids, lengths):
        """
        token_ids: [batch_size, seq_len]
        lengths:   [batch_size]
        返回 logits: [batch_size, 2]
        """
        embedded = self.embedding(token_ids)

        # 打包序列，避免 LSTM 把末尾的 PAD 当成有效文本。
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu().clamp(min=1),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed)

        # 单层双向 LSTM：
        # hidden[-2] 是正向最后一层的隐藏状态；
        # hidden[-1] 是反向最后一层的隐藏状态。
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        features = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1,
        )

        features = self.dropout(features)
        logits = self.classifier(features)

        return logits
