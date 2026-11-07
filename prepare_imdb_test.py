"""Prepare a balanced external movie test set from Stanford's IMDb test split."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import re

import pandas as pd
import requests


SOURCE_URL = (
    "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/"
    "plain_text/test-00000-of-00001.parquet"
)


def prepare_imdb_test(per_class=2000, seed=42):
    if per_class < 1:
        raise ValueError("per_class must be positive")
    raw_path = Path("data/raw/imdb_test.parquet")
    output = Path("data/processed/reviews_IMDB_test.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.is_file():
        partial = raw_path.with_suffix(".parquet.part")
        with requests.get(SOURCE_URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as destination:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        destination.write(chunk)
        # Validate before caching the downloaded source.
        pd.read_parquet(partial, columns=["text", "label"])
        partial.replace(raw_path)

    source = pd.read_parquet(raw_path, columns=["text", "label"])
    if len(source) != 25000 or set(source["label"]) != {0, 1}:
        raise ValueError("Unexpected IMDb official test split")
    source["review"] = source["text"].map(
        lambda text: html.unescape(re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)).strip()
    )
    source = source[source["review"].ne("")]
    # Sample each class separately; do not take the first rows of this ordered dataset.
    selected = pd.concat([
        source[source["label"].eq(label)].sample(n=per_class, random_state=seed)
        for label in (0, 1)
    ]).sample(frac=1, random_state=seed)
    result = selected[["review", "label"]].rename(columns={"label": "voted_up"})
    result["voted_up"] = result["voted_up"].astype("int64")
    result.to_csv(output, index=False, encoding="utf-8")
    metadata = {
        "dataset": "stanfordnlp/imdb",
        "split": "test",
        "source_url": SOURCE_URL,
        "source_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "per_class": per_class,
        "seed": seed,
        "selected_source_row_indices": selected.index.tolist(),
        "label_meanings": {"0": "negative", "1": "positive"},
        "cleaning": "Replace HTML br tags with spaces; decode HTML entities; strip outer whitespace.",
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(result)} IMDb test reviews to {output}")
    print(result["voted_up"].value_counts().sort_index().to_string())
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_imdb_test(args.per_class, args.seed)
