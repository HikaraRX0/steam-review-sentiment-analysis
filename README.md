# Steam Review Sentiment Analysis

Binary sentiment classification for English Steam reviews, using recommendation
labels: **1 = recommended**, **0 = not recommended**. The project covers review
collection, data preprocessing, four baseline models, GloVe initialization
comparisons, FPS versus mixed-game training, and evaluation on games and movie
reviews excluded from training.

## Dataset and Evaluation Design

### FPS Training Dataset

The current cleaned FPS dataset contains **20,172 reviews** from seven games:

| Game | Recommended | Not recommended | Total |
| --- | ---: | ---: | ---: |
| Counter-Strike 2 | 933 | 1,431 | 2,364 |
| Call of Duty: Modern Warfare 4 | 1,688 | 1,676 | 3,364 |
| Apex Legends | 1,102 | 1,489 | 2,591 |
| Battlefield 6 | 1,279 | 1,778 | 3,057 |
| Delta Force | 1,187 | 1,693 | 2,880 |
| Titanfall 2 | 1,548 | 1,779 | 3,327 |
| Rainbow Six Siege | 1,094 | 1,495 | 2,589 |

Reviews are split 80% / 10% / 10%, stratified by game and recommendation label,
with `random_state=42`: **16,137 training**, **2,017 validation**, and
**2,018 internal test** reviews. Thus, the internal test contains the same games
as training, but different review IDs. TF-IDF features and neural vocabularies
are fitted or built from training reviews only.

### Mixed Training Dataset

`MIXED_GAME_IDS` configures a separate dataset containing CS2, GTA V Legacy,
Disco Elysium, Street Fighter 6, Black Myth: Wukong, American Truck Simulator,
and Hades. It does not change `FPS_GAME_IDS`. New training-game
collection requests 2,000 positive and 2,000 negative English reviews per game;
CS2 reuses its existing processed reviews. Cleaned class counts may be lower.

The current cleaned mixed dataset contains **20,346 cleaned reviews**:

| Game | Recommended | Not recommended | Total |
| --- | ---: | ---: | ---: |
| Counter-Strike 2 | 933 | 1,431 | 2,364 |
| GTA V Legacy | 878 | 1,457 | 2,335 |
| Disco Elysium | 1,535 | 1,820 | 3,355 |
| Street Fighter 6 | 1,380 | 1,726 | 3,106 |
| Black Myth: Wukong | 1,308 | 1,743 | 3,051 |
| American Truck Simulator | 1,309 | 1,678 | 2,987 |
| Hades | 1,419 | 1,729 | 3,148 |

The mixed dataset is split into **16,275 training**, **2,034 validation**, and
**2,037 internal test** reviews. These counts are close to, but not identical to,
the original FPS experiment; comparing full-dataset runs also changes sample
composition and training size, rather than isolating genre diversity alone.

### External Test Datasets

Eleven complete game datasets are held out for external evaluation:

| External game | AppID | Recommended | Not recommended | Total | Evaluation purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| WARDOGS | 1867240 | 883 | 943 | 1,826 | Transfer to an unseen FPS game |
| Squad | 393380 | 611 | 853 | 1,464 | Transfer to an unseen tactical FPS |
| Cyberpunk 2077 | 1091500 | 666 | 892 | 1,558 | Transfer to an action RPG |
| Persona 5 Royal | 1687950 | 670 | 857 | 1,527 | Transfer to a turn-based JRPG |
| Baldur's Gate 3 | 1086940 | 729 | 869 | 1,598 | Transfer to another RPG |
| Dota 2 | 570 | 401 | 566 | 967 | Transfer to a MOBA |
| Hollow Knight: Silksong | 1030300 | 659 | 857 | 1,516 | Transfer to a metroidvania |
| Stardew Valley | 413150 | 683 | 762 | 1,445 | Transfer to a farming/life simulation |
| Elden Ring | 1245620 | 641 | 835 | 1,476 | Transfer to an action RPG |
| Forza Horizon 6 | 2483190 | 615 | 917 | 1,532 | Transfer to a racing game |
| Among Us | 945360 | 523 | 778 | 1,301 | Transfer to a social deduction game |

These external reviews are not split again or included in FPS or mixed training,
validation, vocabulary construction, or TF-IDF fitting. The trained models
predict them using their existing text processing. Model settings should be
chosen on the corresponding training dataset's validation set. Internal and
external test sets are reserved for evaluation.

IMDb adds a separate movie-domain test set of **4,000 reviews**: 2,000 positive
and 2,000 negative, sampled from the official test split. Both training conditions
use the same external datasets. Preparation instructions are provided below.

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

Neural models use batch size 64, cross-entropy loss, and Adam. BERT runs for
**5 epochs**; TextCNN and BiLSTM (including GloVe variants) run for **15 epochs**.
The learning rate is `1e-3` for TextCNN/BiLSTM and `2e-5` for BERT. TextCNN and
BiLSTM truncate reviews to 256 tokens; BERT uses a maximum encoded length of 256.
The notebooks set `torch.manual_seed(42)` and seed training-loader shuffling.
The current code evaluates **the final epoch's weights**; automatic best-validation
checkpoint restoration and early stopping are not implemented.

## Recorded Results

The tables below use the external summary outputs currently saved in the four
FPS baseline notebooks and their four mixed-training counterparts. Each entry
is a **single training run**, not an average over random seeds. Both training
conditions use the same external datasets: 11 Steam games and 4,000 IMDb reviews.
Accuracy is shown as a percentage; Macro-F1 gives equal weight to both classes.
Internal FPS and mixed test sets contain different games and are not treated as
matched external comparisons. The first four tables compare FPS and mixed
training; a separate GloVe comparison follows.

### Logistic Regression

| External dataset | FPS accuracy | Mixed accuracy | FPS Macro-F1 | Mixed Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| WARDOGS | 78.64% | 77.88% | 0.7770 | 0.7674 |
| Squad | 84.02% | 83.81% | 0.8298 | 0.8246 |
| Cyberpunk 2077 | 81.84% | 86.33% | 0.8091 | 0.8578 |
| Persona 5 Royal | 78.19% | 86.12% | 0.7741 | 0.8579 |
| Baldur's Gate 3 | 80.91% | 86.61% | 0.8033 | 0.8630 |
| Dota 2 | 78.18% | 78.49% | 0.7628 | 0.7737 |
| Hollow Knight: Silksong | 80.08% | 84.04% | 0.7946 | 0.8350 |
| Stardew Valley | 82.91% | 86.64% | 0.8274 | 0.8659 |
| Elden Ring | 81.44% | 83.74% | 0.8070 | 0.8323 |
| Forza Horizon 6 | 87.14% | 87.34% | 0.8633 | 0.8651 |
| Among Us | 84.24% | 85.55% | 0.8321 | 0.8474 |
| IMDb | 67.05% | 68.00% | 0.6464 | 0.6519 |

### TextCNN

| External dataset | FPS accuracy | Mixed accuracy | FPS Macro-F1 | Mixed Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| WARDOGS | 75.96% | 76.12% | 0.7464 | 0.7511 |
| Squad | 82.79% | 82.51% | 0.8149 | 0.8147 |
| Cyberpunk 2077 | 83.31% | 85.30% | 0.8242 | 0.8476 |
| Persona 5 Royal | 78.26% | 83.30% | 0.7755 | 0.8291 |
| Baldur's Gate 3 | 81.04% | 86.55% | 0.8047 | 0.8631 |
| Dota 2 | 75.59% | 78.18% | 0.7414 | 0.7750 |
| Hollow Knight: Silksong | 80.54% | 82.72% | 0.7978 | 0.8223 |
| Stardew Valley | 80.83% | 84.98% | 0.8061 | 0.8492 |
| Elden Ring | 82.32% | 82.99% | 0.8163 | 0.8264 |
| Forza Horizon 6 | 84.99% | 84.27% | 0.8379 | 0.8316 |
| Among Us | 82.40% | 81.32% | 0.8122 | 0.8074 |
| IMDb | 59.75% | 61.38% | 0.5343 | 0.5581 |

### BiLSTM

| External dataset | FPS accuracy | Mixed accuracy | FPS Macro-F1 | Mixed Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| WARDOGS | 79.08% | 76.62% | 0.7858 | 0.7569 |
| Squad | 82.38% | 83.40% | 0.8146 | 0.8254 |
| Cyberpunk 2077 | 80.87% | 84.66% | 0.8003 | 0.8414 |
| Persona 5 Royal | 77.08% | 83.56% | 0.7640 | 0.8315 |
| Baldur's Gate 3 | 78.79% | 85.61% | 0.7833 | 0.8537 |
| Dota 2 | 77.56% | 77.87% | 0.7649 | 0.7699 |
| Hollow Knight: Silksong | 77.70% | 82.45% | 0.7718 | 0.8204 |
| Stardew Valley | 80.35% | 84.84% | 0.8022 | 0.8480 |
| Elden Ring | 80.15% | 82.59% | 0.7961 | 0.8225 |
| Forza Horizon 6 | 81.53% | 84.79% | 0.8028 | 0.8407 |
| Among Us | 81.86% | 82.71% | 0.8104 | 0.8222 |
| IMDb | 63.08% | 64.98% | 0.6098 | 0.6295 |

### BERT

| External dataset | FPS accuracy | Mixed accuracy | FPS Macro-F1 | Mixed Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| WARDOGS | 90.80% | 90.53% | 0.9077 | 0.9049 |
| Squad | 89.07% | 90.23% | 0.8854 | 0.8988 |
| Cyberpunk 2077 | 88.25% | 91.01% | 0.8787 | 0.9082 |
| Persona 5 Royal | 86.38% | 88.80% | 0.8612 | 0.8869 |
| Baldur's Gate 3 | 90.05% | 91.11% | 0.8995 | 0.9104 |
| Dota 2 | 82.73% | 82.11% | 0.8173 | 0.8148 |
| Hollow Knight: Silksong | 87.73% | 88.46% | 0.8748 | 0.8828 |
| Stardew Valley | 89.97% | 90.17% | 0.8995 | 0.9017 |
| Elden Ring | 87.67% | 87.60% | 0.8740 | 0.8746 |
| Forza Horizon 6 | 90.08% | 90.40% | 0.8957 | 0.8998 |
| Among Us | 87.09% | 85.86% | 0.8652 | 0.8539 |
| IMDb | 86.98% | 87.48% | 0.8695 | 0.8747 |

Mixed training improves the recorded Persona 5 Royal accuracy for all four
models: by 7.93 percentage points for Logistic Regression, 5.04 for TextCNN,
6.48 for BiLSTM, and 2.42 for BERT. Mixed training also improves several other
external-game scores, but the gains are not universal. BERT has the highest
accuracy and Macro-F1 among these four models under both training conditions
on every listed external dataset, including IMDb.

These results are consistent with broader training coverage helping transfer to
some unseen games. They do not identify which training game or type of language
causes an improvement. Small score differences can also reflect training
variation; repeat runs are needed to assess stability. Game-specific class
proportions and review difficulty also affect comparisons across test datasets.

### GloVe Initialization Comparison

The following results compare random embedding initialization with trainable
GloVe 6B 100-dimensional initialization. Both variants use FPS training data,
the same model architecture, and 15 training epochs. GloVe covers **88.0%** of
the training vocabulary (10,214 of 11,613 words); unmatched words retain their
random initialization. Each configuration represents a single training run.

The external-game mean gives equal weight to each of the 11 Steam games.
IMDb is reported separately and is excluded from this mean.

| Model | Test dataset | Baseline accuracy | GloVe accuracy | Baseline Macro-F1 | GloVe Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| TextCNN | FPS internal test | 84.19% | 84.54% | 0.8369 | 0.8403 |
| TextCNN | External games (mean) | 80.73% | 81.64% | 0.7979 | 0.8093 |
| TextCNN | IMDb | 59.75% | 72.38% | 0.5343 | 0.7179 |
| BiLSTM | FPS internal test | 83.65% | 83.05% | 0.8318 | 0.8269 |
| BiLSTM | External games (mean) | 79.76% | 79.91% | 0.7906 | 0.7925 |
| BiLSTM | IMDb | 63.08% | 68.13% | 0.6098 | 0.6731 |

GloVe produces limited overall gains on Steam reviews, with improvements and
regressions across individual games. Mean external-game accuracy increases by
0.91 percentage points for TextCNN and 0.15 for BiLSTM. The IMDb gains are larger:
12.63 percentage points for TextCNN and 5.05 for BiLSTM, accompanied by higher
Macro-F1. These runs suggest that GloVe initialization can help transfer to movie
reviews, but repeated runs are needed to assess stability and statistical tests
are needed before claiming significance.

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
| [config.py](config.py) | Define FPS, mixed, and external game groups |
| [steam_reviews.py](steam_reviews.py) | Collect recommended and not-recommended Steam reviews |
| [clean_data.py](clean_data.py) | Deduplicate by review ID and keep English reviews |
| [prepare_imdb_test.py](prepare_imdb_test.py) | Sample a balanced test set from the official IMDb test split |
| [data_preprocess.ipynb](data_preprocess.ipynb) | Prepare FPS or mixed train, validation, and internal test splits |
| [glove_embeddings.py](glove_embeddings.py) | Download GloVe vectors and initialize training-vocabulary embeddings |
| [text_data.py](text_data.py) / [bert_data.py](bert_data.py) | Prepare datasets and data loaders |
| [train.py](train.py) / [utils.py](utils.py) | Train, evaluate, select devices, and plot training curves |

| Model | FPS notebook | Mixed notebook |
| --- | --- | --- |
| Logistic Regression | [logistic_reg_baseline.ipynb](logistic_reg_baseline.ipynb) | [logistic_reg_mixed.ipynb](logistic_reg_mixed.ipynb) |
| TextCNN | [textcnn_baseline.ipynb](textcnn_baseline.ipynb) | [textcnn_mixed.ipynb](textcnn_mixed.ipynb) |
| BiLSTM | [bilstm_baseline.ipynb](bilstm_baseline.ipynb) | [bilstm_mixed.ipynb](bilstm_mixed.ipynb) |
| BERT | [bert_baseline.ipynb](bert_baseline.ipynb) | [bert_mixed.ipynb](bert_mixed.ipynb) |

GloVe initialization comparisons use [textcnn_glove.ipynb](textcnn_glove.ipynb)
and [bilstm_glove.ipynb](bilstm_glove.ipynb), both trained on FPS splits.

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
MIXED_GAME_IDS = {"CS2": 730, "GTA5": 271590}
EXTERNAL_TEST_GAME_IDS = {
    "WARDOGS": 1867240,
    "CYBERPUNK_2077": 1091500,
    "PERSONA_5_ROYAL": 1687950,
}
GAME_IDS = {**FPS_GAME_IDS, **MIXED_GAME_IDS, **EXTERNAL_TEST_GAME_IDS}
```

`GAME_IDS` is used by the collection script. Set `GENRE` to `"FPS"` or `"MIXED"`
in the preprocessing notebook to select `FPS_GAME_IDS` or `MIXED_GAME_IDS`.
`EXTERNAL_TEST_GAME_IDS` defines held-out games excluded from both FPS and mixed
training and validation. Datasets and downloaded embeddings are excluded from Git;
prepare them using the steps below. No hosted dataset download is provided yet.

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

### Python API

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
listed in the selected training configuration, matching the preprocessing
notebook's expected input.

## Prepare Data and Train Models

### FPS Training

1. Set `GENRE = "FPS"` in `data_preprocess.ipynb` and run it from top to bottom.
   It merges the games in `FPS_GAME_IDS`, converts recommendation labels to 0/1,
   and creates an 80% / 10% / 10% split stratified by game and label.
   The output files are:

   - `data/merged/FPS_train.csv`
   - `data/merged/FPS_val.csv`
   - `data/merged/FPS_test.csv`

2. Run the corresponding FPS notebook from the model table above. Baseline
   notebooks set `GENRE = "FPS"` and load these split files.

### Mixed Training

1. Set `GENRE = "MIXED"` in the same `data_preprocess.ipynb` and run it from top
   to bottom. It merges the games in `MIXED_GAME_IDS` and creates the following
   split files, alongside `data/merged/MIXED_reviews_merged.csv`:

   - `data/merged/MIXED_train.csv`
   - `data/merged/MIXED_val.csv`
   - `data/merged/MIXED_test.csv`

2. Run the corresponding mixed notebook from the model table above. Mixed
   notebooks set `GENRE = "MIXED"` and load these split files.

New mixed games use an approximately 80% / 10% / 10% label-stratified split with
seed 42. CS2 retains the exact assignments in the existing FPS splits, so prepare
FPS splits first. Existing CS2 validation and test reviews do not enter mixed
training. The preprocessing notebook checks review IDs across splits and rejects
overlap between mixed games and external test games. Preparing mixed data does
not overwrite existing FPS split files.

### Training and Evaluation

Run each model notebook from top to bottom. Neural notebooks update training
curves after each completed training and validation epoch. Use validation results
to choose training settings; the internal test cell evaluates the model currently
in memory, which is the final epoch's model after a normal full run. Both FPS and
mixed models use the same external test files.

Train one GPU model at a time and shut down unused notebook kernels to free GPU
memory.

### GloVe Initialization

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
   `FPS_GAME_IDS` or `MIXED_GAME_IDS`, or merge their reviews into the training splits.

3. Prepare the IMDb test CSV using the instructions below before running the
   external evaluation loop.

4. In each trained notebook's existing kernel, run the bottom **External tests**
   section. It loads `review` and `voted_up`, converts Steam `True`/`False` labels
   to `1`/`0`, and also loads the prepared IMDb movie test CSV. It prints metrics
   for each dataset and an `external_results` summary table. The section reloads
   `config.py` so newly configured games are available without restarting the
   trained kernel.

Logistic Regression calls the fitted pipeline's `predict()`. TextCNN/BiLSTM
create test DataLoaders using the training vocabulary; unseen words map to
`<UNK>`. BERT reuses its tokenizer and collate function. Neural evaluation uses
evaluation mode without gradients. No model is fitted on the external reviews.
The trained model and associated vocabulary/tokenizer must still be available in
the kernel; the notebooks do not automatically save and reload model checkpoints.

### IMDb Movie Test Set

Run the following command from the project root:

```sh
python prepare_imdb_test.py --per-class 2000 --seed 42
```

The script downloads the official `test` split from
[stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb), caches it as
`data/raw/imdb_test.parquet`, and samples 2,000 negative and 2,000 positive reviews
without replacement. HTML line-break tags are replaced with spaces and HTML
entities are decoded. No language-detection filter is applied to this existing
English benchmark.

The output `data/processed/reviews_IMDB_test.csv` contains `review` and integer
`voted_up` columns (`0 = negative`, `1 = positive`). Here `voted_up` is a column
name for compatibility: the labels express movie sentiment, not Steam
recommendation decisions. A companion `.metadata.json` records the source URL,
source SHA-256, seed, and selected source row indices. Reading Parquet requires
`pyarrow`, included in `requirements.txt`.

IMDb is kept separate from `EXTERNAL_TEST_GAME_IDS`, which contains Steam AppIDs.
All ten model notebooks load its CSV into `external_test_dfs["IMDB"]` and include
it in the same external evaluation loop using the existing model and
vocabulary/tokenizer. Prepare the CSV before running the **External tests**
section. IMDb is not part of either FPS or mixed training splits.
