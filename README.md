# Steam Review Sentiment Analysis

Binary sentiment classification for English Steam reviews, using recommendation
labels: **1 = recommended**, **0 = not recommended**. The project covers review
collection, data preprocessing, and comparison of four baseline models.

## Project Files

| File | Purpose |
| --- | --- |
| [steam_reviews.py](steam_reviews.py) | Collect positive and negative Steam reviews |
| [clean_data.py](clean_data.py) | Remove duplicate reviews and keep English text |
| [data_preprocess.ipynb](data_preprocess.ipynb) | Merge game datasets, inspect data, and create train/validation/test splits |
| [logistic_reg_baseline.ipynb](logistic_reg_baseline.ipynb) | TF-IDF + Logistic Regression |
| [textcnn_baseline.ipynb](textcnn_baseline.ipynb) | TextCNN classifier |
| [bilstm_baseline.ipynb](bilstm_baseline.ipynb) | Bidirectional LSTM classifier |
| [bert_baseline.ipynb](bert_baseline.ipynb) | Fine-tune `google-bert/bert-base-uncased` |
| `text_data.py` / `bert_data.py` | Dataset and data loader utilities |
| `train.py` / `utils.py` | Shared training, evaluation, device selection, and plotting |

## Setup

Use a Python environment with these dependencies:

```sh
python -m pip install -r requirements.txt
```

Select the same environment as the notebook kernel. For GPU training, use a
PyTorch installation compatible with your CUDA setup. BERT downloads its pretrained
model and tokenizer on the first run.

Create a local `config.py` in the project root with the game names and Steam
AppIDs you want to use. For example:

```python
GAME_IDS = {"CS2": 730}
FPS_GAME_IDS = GAME_IDS.copy()
```

`GAME_IDS` is used by the collection script; `FPS_GAME_IDS` selects the games
merged by the preprocessing notebook. Configuration and datasets are excluded
from Git.

## Collect Reviews

Run commands from the project root. Create `data/raw`, `data/processed`, and
`data/merged` before saving files.

Use a game name from `GAME_IDS` in `config.py`:

```sh
python steam_reviews.py CS2 --num-reviews 500 --language english --output data/raw/reviews_CS2_730_raw.json
```

| Argument | Description | Default |
| --- | --- | --- |
| `game` | Game name from `GAME_IDS`, converted to uppercase | Required |
| `--num-reviews` | Maximum reviews to fetch **per review type** | `100` |
| `--language` | Steam language code, or `all` for all languages | `english` |
| `--output` | Output JSON path | `reviews_<GAME>_<appid>_raw.json` |

The script fetches positive and negative reviews separately, so this example
requests up to **1,000 reviews total**. Fewer may be returned if not enough reviews
are available. Repeat for each configured game.

The output is a JSON list of review objects, including review text, `voted_up`,
language, timestamps, and author information. Positive reviews are saved first,
followed by negative reviews. The terminal reports the counts and output path.

If `--output` is omitted, the JSON is saved in the current directory. The output
directory must already exist; an existing output file is overwritten.

### Use from Python

```python
from steam_reviews import get_reviews

reviews = get_reviews(730, num_reviews=500, review_type=0, language="english")
for review in reviews:
    print(review["review"], review["voted_up"])
```

`review_type=0` includes both types, `1` positive, and `2` negative. A single
function call returns up to `num_reviews` reviews in total, unlike the CLI's
separate requests per type. The function returns data without saving a file.

Requests use pagination with up to 100 reviews per page, a 30-second timeout,
and a one-second delay between pages. The delay can be changed using
`request_delay`; HTTP failures are not automatically retried.

## Clean Reviews

Pass the raw JSON path and the desired CSV path:

```sh
python clean_data.py data/raw/reviews_CS2_730_raw.json data/processed/reviews_CS2_730.csv
```

| Argument | Description |
| --- | --- |
| `input_file` | Raw JSON file produced by the collection tool |
| `output_file` | Destination for the cleaned CSV |

Both arguments are required. Relative paths are resolved from the directory
containing `clean_data.py`; absolute paths also work. The output directory must
already exist, and an existing output file is overwritten.

The tool:

- Extracts `playtime_at_review` from author information.
- Removes duplicate reviews by `recommendationid`.
- Uses `langdetect` on the review text to keep English reviews; text whose
  language cannot be detected is excluded. Short reviews may be misclassified.
- Shuffles rows with `random_state=42` and saves the CSV without a DataFrame index.

The terminal shows recommendation counts before and after language filtering,
followed by the first 10 cleaned rows. The review text itself is preserved.

### Cleaned CSV Columns

| Column | Description |
| --- | --- |
| `recommendationid` | Steam review ID |
| `review` | Original review text |
| `voted_up` | `True` for recommended, `False` for not recommended |
| `language` | Original Steam language value |
| `timestamp_created` | Review creation time as a Unix timestamp |
| `timestamp_updated` | Last update time as a Unix timestamp |
| `playtime_at_review` | Author's playtime at review, in minutes |

Use the filename `reviews_<GAME>_<appid>.csv` in `data/processed` for each game
listed in `FPS_GAME_IDS`, matching the preprocessing notebook's expected input.

## Prepare Data and Train Models

1. Run `data_preprocess.ipynb` from top to bottom. It merges the configured games,
   converts recommendation labels to 0/1, and creates an 80% / 10% / 10% split,
   stratified by game and label:

   - `data/merged/FPS_train.csv`
   - `data/merged/FPS_val.csv`
   - `data/merged/FPS_test.csv`

2. Run a model notebook from top to bottom. Adjust its data paths and training
   parameters as needed. Neural model notebooks update training curves after
   each completed training and validation epoch. Use validation results to choose
   settings, then evaluate the selected model on the held-out test set.

Train one GPU model at a time and shut down unused notebook kernels to free GPU
memory.
