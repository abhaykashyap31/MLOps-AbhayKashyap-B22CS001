# Train Folder Overview

This folder contains modules for training models, including a baseline model and BERT fine-tuning.

## Files and Their Purposes

### `__init__.py`
- Serves as the package initializer for the train module.
- Imports key functions from other modules for easy access.
- Defines `__all__` to specify the public API of the package.

### `baseline.py`
- Implements a baseline model using TF-IDF vectorization and logistic regression.
- Contains `train_baseline` to fit the vectorizer and model on training data.
- Includes `predict_baseline` to make predictions on test data.

### `train_bert.py`
- Handles BERT fine-tuning using DistilBERT for sequence classification.
- Defines `compute_metrics` for evaluation during training.
- Provides `build_trainer` to create a HuggingFace Trainer instance.
- Contains `train_bert` to load the model, configure training, train, and save the model.
- Includes `_save_training_metrics` to log training history to JSON.