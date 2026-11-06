import torch
from torch.utils.data import Dataset

class BertReviewDataset(Dataset):
    """Store review texts and sentiment labels."""
    def __int__(self, df):
        self.texts = df["review"].tolist()
        self.labels = df["voted_up"].astype("int64").tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

def collate_bert_reviews(batch, tokenizer, max_length=256):
    """Tokenize and dynamically pad a batch."""
    texts, labels = zip(* batch)

    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False,
        return_tensors="pt",
    )

    labels = torch.tensor(labels, dtype=torch.long)

    return (encoded["input_ids"], encoded["attention_mask"],), labels
    