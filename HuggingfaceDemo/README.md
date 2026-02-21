# CIFAR-10 Image Classification Demo

## Project Overview

This Jupyter notebook demonstrates image classification on the CIFAR-10 dataset using deep learning techniques. The project showcases data loading from Hugging Face datasets, preprocessing, model training, and evaluation with comprehensive visualizations.

## Features

- **Dataset**: CIFAR-10 subset from Hugging Face
- **Data Augmentation**: Random crop, flip, rotation, color jitter
- **Model Training**: Custom CNN architecture
- **Evaluation**: Accuracy, loss curves, confusion matrix
- **Visualization**: Training progress, model predictions, error analysis

## Dataset

- **Source**: Chiranjeev007/CIFAR-10_Subset
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Split**: Train (5000), Validation (500), Test (1000)
- **Image Size**: 32x32 RGB

## Requirements

- Python 3.8+
- PyTorch
- Torchvision
- Datasets (Hugging Face)
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Installation

```bash
pip install torch torchvision datasets numpy matplotlib seaborn scikit-learn jupyter
```

## Usage

1. Open the notebook:
   ```bash
   jupyter notebook HuggingFace_Demo_B22CS001.ipynb
   ```

2. Run cells sequentially to:
   - Load and explore the dataset
   - Define data transformations
   - Create data loaders
   - Build and train the CNN model
   - Evaluate performance
   - Visualize results

## Model Architecture

Custom CNN with:
- Convolutional layers with batch normalization
- Max pooling
- Dropout for regularization
- Fully connected classifier
- Output: 10 classes

## Training Configuration

- **Optimizer**: Adam or SGD
- **Loss**: Cross-Entropy
- **Batch Size**: Configurable
- **Epochs**: Multiple training epochs
- **Learning Rate**: Adjustable
- **Device**: CPU/GPU automatic detection

## Evaluation Metrics

- Training/Validation accuracy and loss
- Test set performance
- Per-class accuracy
- Confusion matrix
- Classification examples

## Key Components

1. **Data Loading**: Hugging Face datasets integration
2. **Preprocessing**: CIFAR-10 specific normalization
3. **Augmentation**: Training data enhancement
4. **Model Definition**: PyTorch neural network
5. **Training Loop**: With progress tracking
6. **Visualization**: Matplotlib/Seaborn plots
7. **Analysis**: Error case examination

## Results

The notebook provides:
- Model training curves
- Performance metrics
- Sample predictions
- Misclassification analysis
- Comparative visualizations

## File Structure

```
HuggingfaceDemo/
└── HuggingFace_Demo_B22CS001.ipynb  # Main notebook
```

## Key Insights

- Demonstrates complete ML pipeline
- Shows data augmentation benefits
- Illustrates model evaluation techniques
- Provides visualization best practices
- Includes error analysis methods

## Educational Value

This demo serves as:
- Introduction to computer vision with PyTorch
- Hugging Face datasets usage
- CNN implementation guide
- Model training and evaluation tutorial
- Data visualization examples

## Customization

The notebook can be modified for:
- Different datasets
- Alternative architectures
- Hyperparameter tuning
- Additional metrics
- Deployment considerations

## Author

B22CS001

## Note

This is a demonstration notebook for educational purposes, showcasing modern deep learning practices for image classification.</content>
<parameter name="filePath">/home/abhay/Downloads/B22CS001/HuggingfaceDemo/README.md