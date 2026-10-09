# steam-review-sentiment-analysis

Fetch public game reviews through the official
[Steam IUserReviewsService API](https://partner.steamgames.com/doc/webapi/IUserReviewsService).
No API key is required.

```sh
python3 -m pip install -r requirements.txt
python3 steam_reviews.py 570 --num-reviews 500 --language english --output reviews.json
```

Replace `570` with the game's AppID from its Steam store URL. Use `--language all`
for every language. The JSON output preserves review text, author metadata,
timestamps, and `voted_up` (`true` means recommended; `false` means not recommended).
Off-topic reviews are excluded by Steam's default filter.

To use the function from Python:

```python
from steam_reviews import get_reviews

reviews = get_reviews(570, num_reviews=500, language="english")
for review in reviews:
    print(review["review"], review["voted_up"])
```

The fetcher requests up to 100 reviews per page and follows cursors until the
requested count is reached or Steam returns an empty page. It waits one second
between pages. HTTP failures raise an exception; there are no automatic retries.
Results are the newest matching reviews, rather than a random sample.

Run the offline tests with `python3 -m unittest discover -s tests -v`.
