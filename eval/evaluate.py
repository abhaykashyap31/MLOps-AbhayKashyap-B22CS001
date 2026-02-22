def evaluate_trainer(trainer, eval_dataset=None):
    dataset = eval_dataset if eval_dataset is not None else trainer.eval_dataset
    return trainer.evaluate(eval_dataset=dataset)


def get_predictions(trainer, test_dataset, id2label):
    pred = trainer.predict(test_dataset)
    pred_ids = pred.predictions.argmax(-1).flatten().tolist()
    return [id2label[i] for i in pred_ids]
