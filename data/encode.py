from transformers import DistilBertTokenizerFast

from utils.config import MODEL_NAME, MAX_LENGTH


def build_label_mappings(train_labels):
    unique_labels = sorted(set(train_labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def encode_texts_and_labels(
    train_texts,
    train_labels,
    test_texts,
    test_labels,
    tokenizer=None,
    max_length=None,
):
    max_length = max_length or MAX_LENGTH
    if tokenizer is None:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    label2id, id2label = build_label_mappings(train_labels)

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    train_labels_encoded = [label2id[y] for y in train_labels]
    test_labels_encoded = [label2id[y] for y in test_labels]

    return {
        "tokenizer": tokenizer,
        "train_encodings": train_encodings,
        "test_encodings": test_encodings,
        "train_labels_encoded": train_labels_encoded,
        "test_labels_encoded": test_labels_encoded,
        "label2id": label2id,
        "id2label": id2label,
    }
