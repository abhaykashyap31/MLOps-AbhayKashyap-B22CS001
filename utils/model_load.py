from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from utils.config import MODEL_NAME, get_device


def load_tokenizer(model_name=MODEL_NAME):
    """Load the Hugging Face tokenizer for the selected model."""
    return DistilBertTokenizerFast.from_pretrained(model_name)


def load_model_for_classification(model_name=MODEL_NAME, num_labels=8, id2label=None):
    kwargs = {"num_labels": num_labels}
    if id2label is not None:
        kwargs["id2label"] = id2label
    model = DistilBertForSequenceClassification.from_pretrained(model_name, **kwargs)
    device = get_device()
    return model.to(device)
