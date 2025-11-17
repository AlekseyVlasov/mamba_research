import numpy as np
import torch
from torch.utils.data import Dataset


class ParityProblemDataset(Dataset):
    PAD_TOKEN_ID = 0
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    ZERO_TOKEN_ID = 3
    ONE_TOKEN_ID = 4
    PRED_TOKEN_ID = 5
    PRED_ZERO_TOKEN_ID = 6
    PRED_ONE_TOKEN_ID = 7

    def __init__(
        self,
        n_samples: int,
        p: float = 0.5,
        min_seq_len: int = 2,
        max_seq_len: int = 10,
        seed: int = None,
        padding_side: str = "left",
    ):
        super(ParityProblemDataset, self).__init__()
        self.p = p
        self.seed = 0

        # Setting seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Creating sequences
        sequences, attentions = [], []
        for _ in range(n_samples):
            seq_len = np.random.randint(min_seq_len, max_seq_len + 1)
            sequence, attention_mask = self.make_new_sequence(
                seq_len, p, padding_side, max_seq_len
            )
            sequences.append(sequence)
            attentions.append(attention_mask)

        self.sequences = torch.tensor(sequences)
        self.attentions = torch.tensor(attentions)

    def make_new_sequence(self, seq_len, p=0.5, padding_side="left", max_seq_len=10):
        # Creating input sequence
        sequence = np.random.choice(
            [self.ZERO_TOKEN_ID, self.ONE_TOKEN_ID], size=seq_len, p=[1 - p, p]
        )
        solution = (
            ParityProblemDataset.PRED_ZERO_TOKEN_ID
            if np.sum(sequence == ParityProblemDataset.ONE_TOKEN_ID) % 2 == 0
            else ParityProblemDataset.PRED_ONE_TOKEN_ID
        )
        sequence = np.concatenate(
            (
                [ParityProblemDataset.BOS_TOKEN_ID],
                sequence,
                [ParityProblemDataset.PRED_TOKEN_ID],
                [solution],
                [ParityProblemDataset.EOS_TOKEN_ID],
            )
        )
        attention = np.ones_like(sequence, dtype=int)

        if max_seq_len > seq_len:
            padding = (
                np.ones(max_seq_len - seq_len, dtype=int)
                * ParityProblemDataset.PAD_TOKEN_ID
            )
            attention_padding = np.zeros(max_seq_len - seq_len, dtype=int)
            if padding_side == "left":
                sequence = np.concatenate((padding, sequence))
                attention = np.concatenate((attention_padding, attention))
            elif padding_side == "right":
                sequence = np.concatenate((sequence, padding))
                attention = np.concatenate((attention, attention_padding))

        return sequence, attention

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_ids = self.sequences[idx]
        attention_mask = self.attentions[idx]
        labels = torch.ones_like(input_ids) * -100
        labels[input_ids == ParityProblemDataset.PRED_ZERO_TOKEN_ID] = (
            ParityProblemDataset.PRED_ZERO_TOKEN_ID
        )
        labels[input_ids == ParityProblemDataset.PRED_ONE_TOKEN_ID] = (
            ParityProblemDataset.PRED_ONE_TOKEN_ID
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def compute_metrics(predictions):
    logits, labels = predictions
    preds = logits[:, :-1].argmax(axis=-1).flatten()
    labels = labels[:, 1:].flatten()
    preds = preds[labels != -100]
    labels = labels[labels != -100]
    acc = (preds == labels).sum() / len(labels)
    return {
        "accuracy": acc,
    }
