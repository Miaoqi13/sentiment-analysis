
import torch
from torch.utils.data import Dataset

from vocab import encode_text, MAX_SEQ_LEN


class SentimentDataset(Dataset):
    """情感分类数据集。"""

    def __init__(self, dataframe, vocab, max_seq_len=MAX_SEQ_LEN):
        self.reviews = dataframe["review"].astype(str).tolist()
        self.labels = dataframe["label"].astype(int).tolist()
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        review = self.reviews[index]
        label = self.labels[index]

        token_ids = encode_text(
            review,
            self.vocab,
            self.max_seq_len,
        )

        # PAD 的编号为 0，真实长度不包括 PAD。
        length = sum(token_id != 0 for token_id in token_ids)

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )
