import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data import (
    load_or_download_genre_reviews,
    split_train_test,
    encode_texts_and_labels,
    build_datasets,
)
from train.train_bert import build_trainer
from transformers import DistilBertForSequenceClassification, TrainingArguments
from utils.config import get_device
from eval.save_results import run_evaluation_and_save


def main():
    parser = argparse.ArgumentParser(description="Evaluate model from Hugging Face and compare with local")
    parser.add_argument(
        "--repo",
        required=True,
        help="Hugging Face repo id, e.g. username/distilbert-goodreads-genres",
    )
    parser.add_argument(
        "--cache-path",
        default="genre_reviews_dict.pickle",
        help="Pickle cache for genre data",
    )
    parser.add_argument(
        "--local-results",
        default="evaluation_results/evaluation_results.json",
        help="Path to local evaluation_results.json (Task 6)",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Directory for HF eval results and comparison file",
    )
    args = parser.parse_args()

    genre_reviews_dict = load_or_download_genre_reviews(
        cache_path=args.cache_path,
        use_cache=True,
    )
    train_texts, train_labels, test_texts, test_labels = split_train_test(
        genre_reviews_dict,
        reviews_per_genre=1000,
        train_ratio=0.8,
    )
    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.repo)
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
        args.repo,
        num_labels=len(id2label),
    ).to(get_device())
    training_args = TrainingArguments(
        output_dir="./results_hf_eval",
        per_device_eval_batch_size=16,
    )
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=test_dataset,
        eval_dataset=test_dataset,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_results_path = output_dir / "evaluation_results_hf.json"
    hf_results = run_evaluation_and_save(
        trainer,
        test_dataset,
        id2label,
        test_labels,
        hf_results_path,
    )
    print("HF model evaluation saved to {}".format(hf_results_path))
    print("  eval_loss: {:.4f}, accuracy: {:.4f}, f1_macro: {:.4f}, f1_weighted: {:.4f}".format(
        hf_results["eval_loss"],
        hf_results["accuracy"],
        hf_results["f1_macro"],
        hf_results["f1_weighted"],
    ))

    local_path = Path(args.local_results)
    comparison_path = output_dir / "evaluation_comparison.json"
    comparison = {
        "hf_repo": args.repo,
        "local_results_path": str(local_path),
        "hf_results": {
            "eval_loss": hf_results["eval_loss"],
            "accuracy": hf_results["accuracy"],
            "f1_macro": hf_results["f1_macro"],
            "f1_weighted": hf_results["f1_weighted"],
        },
        "local_results": None,
        "comparison": None,
    }
    if local_path.exists():
        with open(local_path) as f:
            local_results = json.load(f)
        comparison["local_results"] = {
            "eval_loss": local_results.get("eval_loss"),
            "accuracy": local_results.get("accuracy"),
            "f1_macro": local_results.get("f1_macro"),
            "f1_weighted": local_results.get("f1_weighted"),
        }
        comparison["comparison"] = {
            "accuracy_diff": hf_results["accuracy"] - local_results.get("accuracy", 0),
            "f1_macro_diff": hf_results["f1_macro"] - local_results.get("f1_macro", 0),
            "f1_weighted_diff": hf_results["f1_weighted"] - local_results.get("f1_weighted", 0),
            "eval_loss_diff": hf_results["eval_loss"] - local_results.get("eval_loss", 0),
        }
        print("Comparison with local model saved to {}".format(comparison_path))
        print("  accuracy diff (HF - local): {:.4f}".format(comparison["comparison"]["accuracy_diff"]))
        print("  f1_macro diff (HF - local): {:.4f}".format(comparison["comparison"]["f1_macro_diff"]))
    else:
        print("Local results not found at {}; comparison contains only HF metrics.".format(local_path))

    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)


if __name__ == "__main__":
    main()
