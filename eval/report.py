import random

from sklearn.metrics import classification_report


def print_classification_report(test_labels, predicted_labels):
    print(classification_report(test_labels, predicted_labels))


def print_sample_predictions(
    test_labels,
    predicted_labels,
    test_texts,
    n_samples=20,
    correct_only=False,
    misclassified_only=False,
):
    triples = list(zip(test_labels, predicted_labels, test_texts))
    if correct_only:
        triples = [t for t in triples if t[0] == t[1]]
    elif misclassified_only:
        triples = [t for t in triples if t[0] != t[1]]

    for true_label, pred_label, text in random.sample(
        triples, min(n_samples, len(triples))
    ):
        if misclassified_only:
            print("TRUE LABEL:", true_label)
            print("PREDICTED LABEL:", pred_label)
        else:
            print("LABEL:", true_label)
        print("REVIEW TEXT:", (text[:200] + "..." if len(text) > 200 else text))
        print()
