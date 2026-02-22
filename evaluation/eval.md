# Eval Folder Overview

This folder contains modules for evaluating the trained model, computing metrics, generating reports, creating plots, and saving results.

## Files and Their Purposes

### `__init__.py`
- Serves as the package initializer for the eval module.
- Imports key functions from other modules for easy access.
- Defines `__all__` to specify the public API of the package.

### `evaluate.py`
- Contains `evaluate_trainer` function to run evaluation on a HuggingFace Trainer and return metrics.
- Includes `get_predictions` function to extract predicted class labels from the trainer's predictions.

### `metrics.py`
- Provides `compute_accuracy_metrics` function to calculate accuracy and generate a classification report using scikit-learn.

### `plots.py`
- Defines helper function `_build_confusion_df` to build a DataFrame for confusion matrices.
- Includes `plot_confusion_heatmap` to plot the full confusion matrix heatmap.
- Contains `plot_misclassification_heatmap` to plot a heatmap of only misclassifications (excluding correct predictions).

### `report.py`
- Contains `print_classification_report` to print the scikit-learn classification report to stdout.
- Includes `print_sample_predictions` to print random samples of predictions, with options to show only correct or misclassified examples.

### `save_results.py`
- Provides `run_evaluation_and_save` function to evaluate the model, compute metrics (accuracy, F1 scores, loss), and save the results to a JSON file, including per-class metrics.