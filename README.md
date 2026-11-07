# Steam Review Sentiment Analysis

Binary sentiment classification for English Steam reviews, using recommendation
labels: **1 = recommended**, **0 = not recommended**. The project covers review
collection, data preprocessing, four baseline models, GloVe initialization
comparisons, and evaluation on games excluded from training.

## Dataset and Evaluation Design

The current cleaned FPS dataset contains **20,172 reviews** from seven games:

| Game | Cleaned reviews |
| --- | ---: |
| Counter-Strike 2 | 2,364 |
| Call of Duty: Modern Warfare 4 | 3,364 |
| Apex Legends | 2,591 |
| Battlefield 6 | 3,057 |
| Delta Force | 2,880 |
| Titanfall 2 | 3,327 |
| Rainbow Six Siege | 2,589 |

Reviews are split 80% / 10% / 10%, stratified by game and recommendation label,
with `random_state=42`: **16,137 training**, **2,017 validation**, and
**2,018 internal test** reviews. Thus, the internal test contains the same games
as training, but different review IDs. TF-IDF features and neural vocabularies
are fitted or built from training reviews only.

Three complete game datasets are held out for external evaluation:

| External game | AppID | Recommended | Not recommended | Total | Evaluation purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| WARDOGS | 1867240 | 883 | 943 | 1,826 | Transfer to an unseen FPS game |
| Cyberpunk 2077 | 1091500 | 666 | 892 | 1,558 | Transfer to an action RPG |
| Persona 5 Royal | 1687950 | 670 | 857 | 1,527 | Transfer to a turn-based JRPG |

These external reviews are not split again or included in FPS training,
validation, vocabulary construction, or TF-IDF fitting. The trained models
predict them using their existing text processing. Model settings should be
chosen on the FPS validation set, rather than on internal or external test scores.

Counts describe the current local data snapshot. Collection requests positive
and negative reviews separately, so these samples do not estimate each game's
natural recommendation rate. Cleaning changes the class proportions. Fetching
again can return different reviews and counts; data files are not included in Git.

## Experiment Settings

| Model | Current configuration |
| --- | --- |
| Logistic Regression | TF-IDF word unigrams/bigrams, `min_df=2`; `C=1.2`, `max_iter=1000` |
| TextCNN | 100-dimensional embeddings; kernels 3/4/5 with 100 channels each; dropout 0.5 before classification |
| BiLSTM | 100-dimensional embeddings; one bidirectional layer, hidden size 100; dropout 0.5 before classification |
| BERT | Fine-tune `google-bert/bert-base-uncased` for two labels |
| GloVe variants | Same TextCNN/BiLSTM settings, with trainable GloVe 6B 100-dimensional embedding initialization |

Neural models use batch size 64, cross-entropy loss, and Adam for 20 epochs.
The learning rate is `1e-3` for TextCNN/BiLSTM and `2e-5` for BERT. TextCNN and
BiLSTM truncate reviews to 256 tokens; BERT uses a maximum encoded length of 256.
The notebooks set `torch.manual_seed(42)` and seed training-loader shuffling.
The current code evaluates **the final epoch's weights**; automatic best-validation
checkpoint restoration and early stopping are not implemented.

## Recorded Results

The following values are transcribed from the saved notebook outputs. Each row
represents a single run, rather than a mean over multiple seeds. Macro-F1 gives
equal weight to the two classes; accuracy is provided as a supplementary metric.

### Macro-F1

| Model | FPS internal test | WARDOGS | Cyberpunk 2077 | Persona 5 Royal |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8488 | 0.7770 | 0.8091 | 0.7741 |
| TextCNN | 0.8367 | 0.7783 | 0.8275 | 0.7784 |
| TextCNN + GloVe | 0.8432 | 0.7893 | 0.8130 | 0.7817 |
| BiLSTM | 0.8363 | 0.7751 | 0.7953 | 0.7578 |
| BiLSTM + GloVe | 0.8266 | 0.7785 | 0.8070 | 0.7669 |
| BERT | **0.8774** | **0.9058** | **0.8864** | **0.8659** |

### Accuracy

| Model | FPS internal test | WARDOGS | Cyberpunk 2077 | Persona 5 Royal |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 85.33% | 78.64% | 81.84% | 78.19% |
| TextCNN | 83.99% | 78.48% | 83.44% | 78.19% |
| TextCNN + GloVe | 84.64% | 79.46% | 82.03% | 78.59% |
| BiLSTM | 83.89% | 77.88% | 80.04% | 76.03% |
| BiLSTM + GloVe | 82.85% | 78.15% | 81.26% | 76.88% |
| BERT | **87.96%** | **90.64%** | **88.96%** | **86.77%** |

BERT has the highest recorded scores on all four test sets. GloVe's gains are
small and vary by model and dataset: it improves all three external BiLSTM
Macro-F1 scores, but lowers its internal score and TextCNN's Cyberpunk score.
These single-run differences do not establish statistical significance. Different
games also have different class proportions and review difficulty, so their
score ordering alone does not establish an effect of genre distance.

### Limitations

- Labels reflect Steam recommendation decisions, which can differ from the
  sentiment of individual sentences in a review.
- Deduplication uses review IDs, not text. Identical text from distinct reviews
  is retained and can occur across splits; this can make internal evaluation
  more optimistic for repeated expressions.
- Language detection can discard valid short English reviews. External class
  counts are unequal after cleaning.
- Neural training curves show substantial train/validation gaps. Results use
  final-epoch weights and have not been averaged over multiple seeds.
- Each external genre is represented by few games. Generalization conclusions
  apply to these sampled games and reviews, rather than to every FPS or RPG.

## Project Files

| File | Purpose |
| --- | --- |
| [steam_reviews.py](steam_reviews.py) | Collect positive and negative Steam reviews |
| [clean_data.py](clean_data.py) | Remove duplicate reviews and keep English text |
| [data_preprocess.ipynb](data_preprocess.ipynb) | Merge game datasets, inspect data, and create train/validation/test splits |
| [logistic_reg_baseline.ipynb](logistic_reg_baseline.ipynb) | TF-IDF + Logistic Regression |
| [textcnn_baseline.ipynb](textcnn_baseline.ipynb) | TextCNN classifier |
| [bilstm_baseline.ipynb](bilstm_baseline.ipynb) | Bidirectional LSTM classifier |
| [textcnn_glove.ipynb](textcnn_glove.ipynb) / [bilstm_glove.ipynb](bilstm_glove.ipynb) | GloVe-initialized variants of the neural baselines |
| `glove_embeddings.py` | Download shared GloVe vectors and initialize training-vocabulary embeddings |
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

Edit `config.py` in the project root with the game names and Steam AppIDs you
want to use. For example:

```python
FPS_GAME_IDS = {"CS2": 730}
RPG_GAME_IDS = {}
EXTERNAL_TEST_GAME_IDS = {
    "WARDOGS": 1867240,
    "CYBERPUNK_2077": 1091500,
    "PERSONA_5_ROYAL": 1687950,
}
GAME_IDS = {**FPS_GAME_IDS, **RPG_GAME_IDS, **EXTERNAL_TEST_GAME_IDS}
```

`GAME_IDS` is used by the collection script; `FPS_GAME_IDS` selects the games
merged by the preprocessing notebook. `EXTERNAL_TEST_GAME_IDS` lists held-out
games for external evaluation and is excluded from FPS training and validation.
Datasets and downloaded embeddings are excluded from Git.

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

1. Run `data_preprocess.ipynb` from top to bottom. It merges games in `FPS_GAME_IDS`,
   converts recommendation labels to 0/1, and creates an 80% / 10% / 10% split,
   stratified by game and label:

   - `data/merged/FPS_train.csv`
   - `data/merged/FPS_val.csv`
   - `data/merged/FPS_test.csv`

2. Run a model notebook from top to bottom. Adjust its data paths and training
   parameters as needed. Neural model notebooks update training curves after
   each completed training and validation epoch. Use validation results to choose
   settings. The test cell evaluates the model currently in memory, which is the
   final epoch's model after running the notebook normally.

Train one GPU model at a time and shut down unused notebook kernels to free GPU
memory.

The GloVe notebooks preserve the baseline architecture and training settings, but
initialize matching embedding rows with Stanford GloVe 6B 100-dimensional vectors.
Run them from the top in separate kernels and train fresh models. The first run
downloads the official approximately 862 MB archive to `data/embeddings` and
extracts `glove.6B.100d.txt`; both notebooks reuse these files. Embeddings are
fine-tuned by default (`freeze_embeddings = False`). Unmatched words keep their
random initialization, padding stays zero, and vocabulary is still built from
FPS training data only. Each notebook reports GloVe vocabulary coverage and
retains the existing internal and external test sections.

## Run External Tests

1. Collect and clean each game in `EXTERNAL_TEST_GAME_IDS` using the same scripts.
   For example:

   ```sh
   python steam_reviews.py WARDOGS --num-reviews 1000 --language english --output data/raw/reviews_WARDOGS_1867240_raw.json
   python clean_data.py data/raw/reviews_WARDOGS_1867240_raw.json data/processed/reviews_WARDOGS_1867240.csv
   ```

2. Keep the filename pattern `data/processed/reviews_<GAME>_<appid>.csv`. These
   files are the complete external test sets; do not add the games to
   `FPS_GAME_IDS` or merge their reviews into the training splits.

3. In each trained notebook's existing kernel, run the bottom **External game
   tests** section. It loads `review` and `voted_up`, converts `True`/`False` to
   `1`/`0`, and prints per-game metrics and an `external_results` summary table.

Logistic Regression calls the fitted pipeline's `predict()`. TextCNN/BiLSTM
create test DataLoaders using the training vocabulary; unseen words map to
`<UNK>`. BERT reuses its tokenizer and collate function. Neural evaluation uses
evaluation mode without gradients. No model is fitted on the external reviews.
The trained model and associated vocabulary/tokenizer must still be available in
the kernel; the notebooks do not automatically save and reload model checkpoints.
