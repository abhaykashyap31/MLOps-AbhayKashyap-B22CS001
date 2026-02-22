# Data Folder Overview

This folder contains modules for handling data loading, preprocessing, encoding, and dataset creation for the book genre classification project.

## Files and Their Purposes

### `__init__.py`
- Serves as the package initializer for the data module.
- Imports key functions and classes from other modules for easy access.
- Defines `__all__` to specify the public API of the package.

### `dataset.py`
- Defines the `ReviewDataset` class, a PyTorch Dataset subclass for handling encoded BERT inputs and labels.
- Includes the `build_datasets` function to create train and test datasets from encoded data.

### `encode.py`
- Contains `build_label_mappings` function to create label-to-ID and ID-to-label mappings from training labels.
- Includes `encode_texts_and_labels` function to tokenize texts using DistilBERT tokenizer and encode labels to integers.

### `load.py`
- Provides `load_reviews` function to stream and sample reviews from gzipped JSON URLs.
- Includes `load_or_download_genre_reviews` function to load reviews per genre, with optional caching to a pickle file.

### `split.py`
- Contains `split_train_test` function to split genre-based reviews into train and test lists based on specified ratios and sample sizes.