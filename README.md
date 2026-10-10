# Steam Review Sentiment Analysis

Collect and clean public Steam game reviews for sentiment analysis using the official
[Steam IUserReviewsService API](https://partner.steamgames.com/doc/webapi/IUserReviewsService).
`steam_reviews.py` fetches raw reviews, and `clean_data.py` prepares a CSV dataset
for analysis. The scripts do not train a model or predict sentiment. No API key
is required.

## Installation

Install Python 3, create a virtual environment, and install the dependencies
(`requests`, `langdetect`, and `pandas`). Run these commands from the project directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Activate the environment with `source .venv/bin/activate` in each new terminal
session before running the scripts. On Windows, use `.venv\Scripts\activate`
instead. If using an IDE, select the Python interpreter inside `.venv`.

## Collect Reviews

```sh
python steam_reviews.py 570 --num-reviews 500 --language english
```

Replace `570` with the game's AppID from its Steam store URL.

| Argument | Description | Default |
| --- | --- | --- |
| `appid` | Steam game AppID (required) | — |
| `--num-reviews` | Maximum number of reviews to fetch **per review type** | `100` |
| `--language` | Steam language code, or `all` for every language | `english` |

The script fetches positive and negative reviews separately. The example above
requests up to 500 of each, for up to 1,000 reviews total. Fewer reviews are saved
if Steam has fewer matching reviews available.

### Raw JSON Output

Reviews are saved to `reviews_<appid>_raw.json` in the current directory, for
example `reviews_570_raw.json`. Running the script again for the same AppID
overwrites this file.

The JSON file contains a list of raw review objects, including:

- `review`: Review text.
- `voted_up`: `true` for recommended, `false` for not recommended.
- `language`: Language of the review.
- `timestamp_created` and `timestamp_updated`: Unix timestamps.
- `author`: Author metadata, including playtime.

Positive reviews appear first, followed by negative reviews. Each group is
fetched newest first. Off-topic reviews are excluded by Steam's default filter.

## Clean Reviews

Run `clean_data.py` with the input JSON path followed by the output CSV path:

```sh
python clean_data.py reviews_570_raw.json reviews_570.csv
```

For the sample dataset in this project:

```sh
python clean_data.py reviews_730_raw.json reviews_730.csv
```

| Argument | Description |
| --- | --- |
| `input_file` | Path to the raw JSON dataset (required) |
| `output_file` | Path to the cleaned CSV dataset (required) |

Both paths can be customized. Relative paths are resolved from the directory
containing `clean_data.py`; absolute paths are also accepted. The output directory
must already exist. An existing output file is overwritten.

The cleaning steps include:

- Extracting `playtime_at_review` from each review's `author` metadata.
- Keeping the columns listed below.
- Removing duplicate reviews by `recommendationid`.
- Detecting the language of the review text with `langdetect` and keeping reviews
  detected as English. Reviews whose language cannot be detected are removed.

Language filtering uses the review text, rather than the JSON `language` field.
Short or ambiguous reviews may be misclassified. A fixed detection seed makes
the results reproducible.

The terminal prints positive (`True`) and negative (`False`) review counts before
and after language filtering, along with a preview of the first 10 output rows.
These messages report processing results and are not errors.

### Cleaned CSV Output

The CSV is saved without a DataFrame index and contains:

| Column | Description |
| --- | --- |
| `recommendationid` | Steam review ID |
| `review` | Original review text |
| `voted_up` | `True` for recommended, `False` for not recommended |
| `language` | Original language value from Steam |
| `timestamp_created` | Review creation time as a Unix timestamp |
| `timestamp_updated` | Last update time as a Unix timestamp |
| `playtime_at_review` | Author's playtime at review, in minutes |

The script preserves review text; it does not tokenize it or remove punctuation.

## Python Example

```python
from steam_reviews import get_reviews

reviews = get_reviews(
    appid=570,
    num_reviews=500,
    review_type=0,
    language="english",
)

for review in reviews:
    print(review["review"], review["voted_up"])
```

Use `review_type=0` for both types, `1` for positive reviews, or `2` for negative
reviews. Unlike the command-line script, a single function call returns up to
`num_reviews` reviews in total.

The fetcher uses cursor pagination with up to 100 reviews per request, a
30-second request timeout, and a one-second delay between pages. Adjust the
delay with `request_delay`. HTTP failures raise an exception; requests are not
automatically retried.
