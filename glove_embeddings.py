"""Download Stanford GloVe and initialize an existing training-only vocabulary."""

from pathlib import Path
import shutil
from zipfile import ZipFile

import requests
import torch


GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_FILE = "glove.6B.100d.txt"


def ensure_glove_100d(data_dir="data/embeddings"):
    """Cache the official archive and extract only its 100-dimensional vectors."""
    data_dir = Path(data_dir)
    vectors = data_dir / GLOVE_FILE
    if vectors.is_file():
        return vectors

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "glove.6B.zip"
    if not archive.is_file():
        partial = archive.with_suffix(".zip.part")
        print("Downloading Stanford glove.6B.zip (about 862 MB); cached for both notebooks.", flush=True)
        downloaded = 0
        reported = 0
        with requests.get(GLOVE_URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        downloaded += len(chunk)
                        if downloaded - reported >= 64 * 1024 * 1024:
                            print(f"Downloaded {downloaded / 1024**2:.0f} MiB", flush=True)
                            reported = downloaded
        # Do not cache an incomplete or unexpected archive as a successful download.
        with ZipFile(partial) as zipped:
            zipped.getinfo(GLOVE_FILE)
        partial.replace(archive)

    partial_vectors = vectors.with_suffix(".txt.part")
    with ZipFile(archive) as zipped:
        with zipped.open(GLOVE_FILE) as source, partial_vectors.open("wb") as output:
            shutil.copyfileobj(source, output)
    partial_vectors.replace(vectors)
    return vectors


def initialize_glove_embedding(embedding, vocab, vectors_path, *, freeze=False):
    """Replace matched rows, preserving random OOV rows and a zero padding row.

    Uses only the supplied vocabulary; never adds words from validation/test data.
    Call after model initialization and before constructing the training optimizer.
    """
    if len(vocab) != embedding.num_embeddings or set(vocab.values()) != set(range(len(vocab))):
        raise ValueError("Vocabulary IDs must match the embedding rows.")

    weights = embedding.weight.detach().cpu().clone()
    wanted = {word: index for word, index in vocab.items() if word not in ("<PAD>", "<UNK>")}
    matched = set()
    with Path(vectors_path).open(encoding="utf-8") as source:
        for line in source:
            word, separator, values = line.rstrip().partition(" ")
            if word not in wanted:
                continue
            components = values.split()
            if not separator or len(components) != embedding.embedding_dim:
                raise ValueError(f"Wrong GloVe dimension for {word!r}; expected {embedding.embedding_dim}.")
            vector = torch.tensor([float(value) for value in components], dtype=weights.dtype)
            if not torch.isfinite(vector).all():
                raise ValueError(f"Non-finite GloVe vector for {word!r}.")
            weights[wanted[word]] = vector
            matched.add(word)

    if not matched:
        raise ValueError("No training vocabulary words matched the GloVe file.")
    if embedding.padding_idx is not None:
        weights[embedding.padding_idx].zero_()
    with torch.no_grad():
        embedding.weight.copy_(weights.to(embedding.weight.device))
    embedding.weight.requires_grad_(not freeze)

    total = len(wanted)
    print(f"GloVe coverage: {len(matched)}/{total} vocabulary words ({len(matched) / total:.1%})")
    print(f"Embedding trainable: {not freeze}; unmatched words retain their random initialization.")
    return {"matched": len(matched), "total": total, "coverage": len(matched) / total}
