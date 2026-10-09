# Steam Review Sentiment Analysis

Collect public Steam game reviews for sentiment analysis using the official
[Steam IUserReviewsService API](https://partner.steamgames.com/doc/webapi/IUserReviewsService).
The current script fetches and saves raw reviews; it does not train a model or
predict sentiment. No API key is required.

## Installation

Install Python 3 and the required dependency:

```sh
python3 -m pip install -r requirements.txt
```

## Usage

```sh
python3 steam_reviews.py 570 --num-reviews 500 --language english
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

## Output

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
