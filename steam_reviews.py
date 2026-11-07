"""Fetch public Steam reviews using IUserReviewsService.GetAppReviews."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import requests

from config import GAME_IDS, FPS_GAME_IDS, MIXED_GAME_IDS, EXTERNAL_TEST_GAME_IDS


API_URL = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"


def get_reviews(appid, num_reviews, review_type=0, language="english", *, request_delay=1.0):
    """Fetch the newest reviews; language="all" includes every language."""
    if num_reviews <= 0:
        raise ValueError("num_reviews must be positive")
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


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def select_games(game, groups):
    if game:
        return {game: GAME_IDS[game]}
    configurations = {
        "FPS": FPS_GAME_IDS,
        "MIXED": MIXED_GAME_IDS,
        "EXTERNAL": EXTERNAL_TEST_GAME_IDS,
        "ALL": GAME_IDS,
    }
    selected = {}
    seen_appids = set()
    for group in groups:
        for name, appid in configurations[group].items():
            if appid not in seen_appids:
                selected[name] = appid
                seen_appids.add(appid)
    return selected


def fetch_game(game, appid, count, language, output):
    positive = get_reviews(appid, count, 1, language)
    negative = get_reviews(appid, count, 2, language)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Only replace the destination after both requests and serialization succeed.
    payload = json.dumps(positive + negative, ensure_ascii=False, indent=2)
    partial = output.with_suffix(output.suffix + ".part")
    partial.write_text(payload, encoding="utf-8")
    partial.replace(output)
    print(f"[{game}] Fetched {len(positive)} positive and {len(negative)} negative reviews: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game", nargs="?", type=str.upper, choices=GAME_IDS, help="Single game name")
    parser.add_argument("--group", nargs="+", type=str.upper,
                        choices=("FPS", "MIXED", "EXTERNAL", "ALL"),
                        help="Batch groups; shared AppIDs are fetched once")
    parser.add_argument("--num-reviews", type=positive_int, required=True,
                        help="Required maximum reviews per type, per game")
    parser.add_argument("--language", default="english", help="Steam language code or all")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Batch output directory (default: data/raw)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing raw and cleaned files")
    parser.add_argument("--clean", action="store_true", help="Run clean_data.py after collecting each game")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"),
                        help="Cleaned CSV directory (default: data/processed)")
    args = parser.parse_args()
    if bool(args.game) == bool(args.group):
        parser.error("choose either a game or --group")
    if args.group and args.output:
        parser.error("--output is for a single game; use --output-dir for batches")
    if args.game and args.output_dir is not None:
        parser.error("--output-dir is for batches; use --output for a single game")

    games = select_games(args.game, args.group)
    print(f"Selected {len(games)} games; requesting up to {args.num_reviews} reviews per type per game.")
    failures = []
    for game, appid in games.items():
        filename = f"reviews_{game}_{appid}_raw.json"
        output = ((args.output_dir or Path("data/raw")) / filename if args.group
                  else Path(args.output or filename))
        try:
            if output.exists() and not args.overwrite:
                print(f"[{game}] Skipped existing raw file: {output}")
            else:
                fetch_game(game, appid, args.num_reviews, args.language, output)
            if args.clean:
                processed = args.processed_dir / f"reviews_{game}_{appid}.csv"
                if processed.exists() and not args.overwrite:
                    print(f"[{game}] Skipped existing cleaned file: {processed}")
                else:
                    processed.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run([
                        sys.executable, str(Path(__file__).with_name("clean_data.py").resolve()),
                        str(output.resolve()), str(processed.resolve()),
                    ], check=True)
        except (requests.RequestException, ValueError, KeyError, TypeError,
                OSError, subprocess.CalledProcessError) as exc:
            failures.append(game)
            print(f"[{game}] Failed: {exc}", file=sys.stderr)

    print(f"Finished: {len(games) - len(failures)} succeeded or skipped; {len(failures)} failed.")
    if failures:
        parser.exit(1, "Failed games: " + ", ".join(failures) + "\n")


if __name__ == "__main__":
    main()
