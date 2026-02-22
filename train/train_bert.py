import json
import os
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from utils.config import MODEL_NAME, CACHED_MODEL_DIR, get_device

os.environ.setdefault("WANDB_DISABLED", "true")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def build_trainer(
    model,
    training_args,
    train_dataset,
    eval_dataset,
    compute_metrics_fn=None,
):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn or compute_metrics,
    )


def train_bert(
    train_dataset,
    test_dataset,
    id2label,
    model_name=MODEL_NAME,
    output_dir="./results",
    logging_dir="./logs",
    num_train_epochs=3,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=None,
    save_dir=None,
):
    device = get_device()
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
    ).to(device)

    if eval_steps is None:
        eval_steps = logging_steps

    training_args = TrainingArguments(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        output_dir=output_dir,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        report_to=[],
    )

    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics_fn=compute_metrics,
    )

    trainer.train()

    save_dir = save_dir or CACHED_MODEL_DIR
    trainer.save_model(save_dir)

    _save_training_metrics(trainer, output_dir)

    return trainer, model


def _save_training_metrics(trainer, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_metrics.json"
    log_history = []
    for entry in trainer.state.log_history:
        clean = {}
        for k, v in entry.items():
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                clean[k] = v
        log_history.append(clean)
    with open(path, "w") as f:
        json.dump(log_history, f, indent=2)
