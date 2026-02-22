# Scripts Folder Overview

This folder contains standalone scripts for various tasks related to model deployment, evaluation, and comparison.

## Files and Their Purposes

### `evaluate_from_hf.py`
- Script to load a model from a Hugging Face repository, run evaluation on the test dataset, and compare the results with local evaluation metrics.
- Outputs evaluation results and a comparison JSON file.

### `push_to_huggingface.py`
- Script to push the fine-tuned model, tokenizer, and configuration to a Hugging Face repository.
- Requires Hugging Face authentication token and handles repo creation.

### `run_eval_only.py`
- Evaluation-only script designed for Docker containers.
- Loads a model from Hugging Face using environment variables, runs evaluation, and saves results to a specified output directory.