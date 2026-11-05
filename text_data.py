import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_ID = 0  # Padding token ID
UNK_ID = 1  # Unknown token ID

def tokenize(text):
    """Tokenize lowercase text, preserving punctuation and straight apostrophes within words."""
    return re.findall(r"\w+(?:[']\w+)*|[^\w\s]", text.lower())

def build_vocab(texts, min_freq=2, max_size=20000):
    """Build a frequency-filtered vocabulary with reserved PAD and UNK IDs."""
    counts = Counter(
        token 
        for text in texts
        for token in tokenize(text)
    )

    vocab = {"<PAD>": PAD_ID, "<UNK>":UNK_ID}

    for token, count in counts.most_common(max_size - 2):
        if count >= min_freq:
            vocab[token] = len(vocab)

    return vocab

# # Load processed training data
# train_df = pd.read_csv("data/merged/FPS_train.csv")

# # Calculate token lengths
# lengths = train_df["review"].apply(
#     lambda text: len(tokenize(text))
# )

# # Show token length distribution
# print(lengths.describe(
#     percentiles=[0.5, 0.9, 0.95, 0.99]
# ))

class ReviewDataset(Dataset):
    """Store truncated review token ID sequences and sentiment labels."""

    def __init__(self, df, vocab, max_length=256):
        self.samples = []

        for text, label in zip(df["review"],df["voted_up"]):
            # Truncate long reviews.
            tokens = tokenize(text)[:max_length]

            # Use UNK for unknown words and as a fallback for empty sequences.
            ids = [
                vocab.get(token, UNK_ID)
                 for token in tokens
            ] or [UNK_ID]

            self.samples.append((
                torch.tensor(ids, dtype=torch.long),
                int(label),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_reviews(batch):
    """Pad a batch and return token IDs, valid lengths, and labels."""
    sequences, labels = zip(*batch)

    # Record valid lengths before padding.
    lengths = torch.tensor(
        [len(seq) for seq in sequences],
        dtype=torch.long
    )

    # Pad to the longest sequence in this batch: [batch_size, sequence_length].
    token_ids = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=PAD_ID,
    )

    # Ensure at least 5 positions for TextCNN kernels up to size 5.
    if token_ids.shape[1] < 5:
        token_ids = F.pad(
            token_ids,
            (0, 5 - token_ids.shape[1]),
            value=PAD_ID,
        )

    labels = torch.tensor(labels, dtype=torch.long)

    return (token_ids, lengths), labels

def create_dataloader(data_dir, genre, batch_size=32):
    """Return train, validation, and test DataLoaders plus a training-only vocabulary."""
    data_dir = Path(data_dir)

    # Read the training, validation, and test splits.
    frames={}
    for split in ("train", "val", "test"):
        frames[split] = pd.read_csv(
            data_dir / f"{genre}_{split}.csv",
            dtype={"review": "string", "voted_up": "int64"}
        )

    # Build the vocabulary from training data only.
    vocab = build_vocab(frames["train"]["review"])

    # Shuffle training samples and use collate_fn to assemble batches.
    loaders={}
    for split, df in frames.items():
        dataset = ReviewDataset(df, vocab)

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_reviews,
            num_workers=0,
            # Seed training data shuffling for reproducibility.
            generator=(
                torch.Generator().manual_seed(42)
                if split == "train" else None
            )
        )

    return loaders["train"], loaders["val"], loaders["test"], vocab

