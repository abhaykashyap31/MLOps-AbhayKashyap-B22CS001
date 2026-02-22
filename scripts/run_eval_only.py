import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

REPO = os.environ.get("HF_REPO")
if not REPO:
    print("Set HF_REPO (e.g. username/distilbert-goodreads-genres) to run evaluation.")
    sys.exit(1)

OUTPUT_DIR = Path(os.environ.get("EVAL_OUTPUT_DIR", "/results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    from data import (
        load_or_download_genre_reviews,
        split_train_test,
        encode_texts_and_labels,
        build_datasets,
    )
    from train.train_bert import build_trainer
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
    from transformers import TrainingArguments
    from utils.config import get_device
    from evaluation.save_results import run_evaluation_and_save

    cache_path = os.environ.get("CACHE_PATH", "genre_reviews_dict.pickle")
    genre_reviews_dict = load_or_download_genre_reviews(
        cache_path=cache_path,
        use_cache=True,
    )
    train_texts, train_labels, test_texts, test_labels = split_train_test(
        genre_reviews_dict,
        reviews_per_genre=1000,
        train_ratio=0.8,
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(REPO)
    encoded = encode_texts_and_labels(
        train_texts,
        train_labels,
        test_texts,
        test_labels,
        tokenizer=tokenizer,
    )
    _, test_dataset = build_datasets(encoded)
    id2label = encoded["id2label"]

    model = DistilBertForSequenceClassification.from_pretrained(
        REPO,
        num_labels=len(id2label),
    ).to(get_device())
    training_args = TrainingArguments(
        output_dir="/tmp/results_eval",
        per_device_eval_batch_size=16,
    )
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=test_dataset,
        eval_dataset=test_dataset,
    )

    results_path = OUTPUT_DIR / "evaluation_results.json"
    results = run_evaluation_and_save(
        trainer,
        test_dataset,
        id2label,
        test_labels,
        results_path,
    )
    print("Evaluation complete. Results written to {}".format(results_path))
    print(json.dumps({k: v for k, v in results.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
