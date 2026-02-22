import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data import (
    load_or_download_genre_reviews,
    split_train_test,
    encode_texts_and_labels,
    build_datasets,
)
from train import train_baseline, predict_baseline, train_bert
from eval import (
    evaluate_trainer,
    get_predictions,
    print_classification_report,
    print_sample_predictions,
    plot_confusion_heatmap,
    plot_misclassification_heatmap,
    run_evaluation_and_save,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Goodreads BERT fine-tuning pipeline")
    parser.add_argument(
        "--cache-path",
        default="genre_reviews_dict.pickle",
        help="Path to pickle cache for genre reviews",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cache and re-download data",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip TF-IDF + logistic regression baseline",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip BERT fine-tuning (evaluate only with existing model)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not show confusion heatmaps",
    )
    parser.add_argument(
        "--results-dir",
        default="evaluation_results",
        help="Directory for evaluation and training metrics (Task 6)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    genre_reviews_dict = load_or_download_genre_reviews(
        cache_path=args.cache_path,
        use_cache=not args.no_cache,
    )

    # Split
    train_texts, train_labels, test_texts, test_labels = split_train_test(
        genre_reviews_dict,
        reviews_per_genre=1000,
        train_ratio=0.8,
    )
    print(
        f"Train: {len(train_texts)} samples, Test: {len(test_texts)} samples"
    )

    # Baseline
    if not args.skip_baseline:
        vectorizer, baseline_model = train_baseline(train_texts, train_labels)
        baseline_preds = predict_baseline(vectorizer, baseline_model, test_texts)
        print("Baseline (TF-IDF + Logistic Regression) classification report:")
        print_classification_report(test_labels, baseline_preds)

    # Encode for BERT
    encoded = encode_texts_and_labels(
        train_texts,
        train_labels,
        test_texts,
        test_labels,
    )
    train_dataset, test_dataset = build_datasets(encoded)
    id2label = encoded["id2label"]

    # BERT fine-tuning
    if not args.skip_train:
        trainer, _ = train_bert(
            train_dataset,
            test_dataset,
            id2label,
        )
        # Save tokenizer to model dir so Task 7 (push to HF) includes it
        encoded["tokenizer"].save_pretrained(CACHED_MODEL_DIR)
    else:
        # Load saved model and build a minimal Trainer for evaluation only
        from utils.config import CACHED_MODEL_DIR, get_device
        from transformers import DistilBertForSequenceClassification
        from train.train_bert import build_trainer
        from transformers import TrainingArguments

        model = DistilBertForSequenceClassification.from_pretrained(
            CACHED_MODEL_DIR,
            num_labels=len(id2label),
        ).to(get_device())
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_eval_batch_size=16,
        )
        trainer = build_trainer(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

    # Evaluate (Task 6: record accuracy, F1, loss and save)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_path = results_dir / "evaluation_results.json"
    eval_results = run_evaluation_and_save(
        trainer,
        test_dataset,
        id2label,
        test_labels,
        eval_path,
    )
    print("BERT eval metrics (saved to {}):".format(eval_path))
    print("  eval_loss: {:.4f}, accuracy: {:.4f}, f1_macro: {:.4f}, f1_weighted: {:.4f}".format(
        eval_results["eval_loss"],
        eval_results["accuracy"],
        eval_results["f1_macro"],
        eval_results["f1_weighted"],
    ))

    metrics = evaluate_trainer(trainer, test_dataset)
    predicted_labels = get_predictions(trainer, test_dataset, id2label)
    print("Fine-tuned BERT classification report:")
    print_classification_report(test_labels, predicted_labels)

    # Sample predictions
    print("Sample correct predictions:")
    print_sample_predictions(
        test_labels,
        predicted_labels,
        test_texts,
        n_samples=5,
        correct_only=True,
    )
    print("Sample misclassifications:")
    print_sample_predictions(
        test_labels,
        predicted_labels,
        test_texts,
        n_samples=5,
        misclassified_only=True,
    )

    # Plots
    if not args.no_plots:
        plot_confusion_heatmap(test_labels, predicted_labels)
        plot_misclassification_heatmap(test_labels, predicted_labels)


if __name__ == "__main__":
    main()
