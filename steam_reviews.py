"""Fetch public Steam reviews using IUserReviewsService.GetAppReviews."""

import argparse
import json
from pathlib import Path
import time

import requests


API_URL = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"


def get_reviews(appid, num_reviews=100, review_type=0, language="english", *, request_delay=1.0):
    """Fetch the newest reviews; language="all" includes every language."""
    reviews = []
    cursor = "*"

    while len(reviews) < num_reviews:
        input_data = {
            "appid": appid,
            "filter": 1,  # Sort by newest first.
            "languages": [language],
            "review_type": review_type,  # Include reviews (0: positive and negative).
            "purchase_type": 1,  # Include all purchase sources.
            "num_per_page": min(100, num_reviews - len(reviews)),
            "cursor": cursor,
        }
        response = requests.get(
            API_URL,
            params={"input_json": json.dumps(input_data)},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()["response"]
        page = data.get("reviews", [])
        if not page:
            break

        reviews.extend(page)
        if len(reviews) >= num_reviews:
            break

        next_cursor = data.get("cursor")
        if not next_cursor or next_cursor == cursor:
            raise ValueError("Steam did not return a new pagination cursor")
        cursor = next_cursor
        time.sleep(request_delay)

    return reviews[:num_reviews]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("appid", type=int, help="Steam game AppID (e.g. 570)")
    parser.add_argument("--num-reviews", type=int, default=100)
    parser.add_argument("--language", default="english", help="Steam language code or all")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    try:
        # Fetch positive reviews
        positive_reviews = get_reviews(
            args.appid,
            args.num_reviews,
            1,
            args.language
        )

        # Fetch negative reviews
        negative_reviews = get_reviews(
            args.appid,
            args.num_reviews,
            2,
            args.language
        )
        reviews = positive_reviews + negative_reviews

        output = Path(args.output or f"reviews_{args.appid}_raw.json")
        output.write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
    except (requests.RequestException, ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Successfully fetched {len(positive_reviews)} positive reviews")
    print(f"Successfully fetched {len(negative_reviews)} negative reviews")
    print(f"Saved {len(reviews)} total reviews to {output}")


if __name__ == "__main__":
    main()
