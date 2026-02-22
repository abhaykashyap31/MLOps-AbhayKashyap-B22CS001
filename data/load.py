import gzip
import json
import pickle
import random
from pathlib import Path

import requests

from utils.config import GENRE_URL_DICT


def load_reviews(url, head=10000, sample_size=2000):
    reviews = []
    count = 0
    response = requests.get(url, stream=True)
    with gzip.open(response.raw, "rt", encoding="utf-8") as file:
        for line in file:
            obj = json.loads(line)
            reviews.append(obj["review_text"])
            count += 1
            if head is not None and count >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))


def load_or_download_genre_reviews(
    genre_url_dict=None,
    head=10000,
    sample_size=2000,
    cache_path="genre_reviews_dict.pickle",
    use_cache=True,
):
    genre_url_dict = genre_url_dict or GENRE_URL_DICT
    cache_path = Path(cache_path)

    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    genre_reviews_dict = {}
    for genre, url in genre_url_dict.items():
        print(f"Loading reviews for genre: {genre}")
        genre_reviews_dict[genre] = load_reviews(url, head=head, sample_size=sample_size)

    with open(cache_path, "wb") as f:
        pickle.dump(genre_reviews_dict, f)

    return genre_reviews_dict
