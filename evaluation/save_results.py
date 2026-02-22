import json
from pathlib import Path


def run_evaluation_and_save(trainer, test_dataset, id2label, test_labels, output_path):
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    from evaluation.evaluate import get_predictions

    metrics = trainer.evaluate(eval_dataset=test_dataset)
    predicted_labels = get_predictions(trainer, test_dataset, id2label)

    f1_macro = f1_score(test_labels, predicted_labels, average="macro", zero_division=0)
    f1_weighted = f1_score(
        test_labels, predicted_labels, average="weighted", zero_division=0
    )
    acc = metrics.get("eval_accuracy")
    if acc is None:
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(test_labels, predicted_labels)

    results = {
        "eval_loss": float(metrics.get("eval_loss", 0)),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }
    labels_sorted = sorted(id2label.values())
    p, r, f1, _ = precision_recall_fscore_support(
        test_labels, predicted_labels, labels=labels_sorted, zero_division=0
    )
    results["per_class"] = {
        label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])}
        for i, label in enumerate(labels_sorted)
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
