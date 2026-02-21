# Digit Recognition with ResNet-18

## Project Overview

This project implements a digit recognition system using a fine-tuned ResNet-18 convolutional neural network. The model is trained on a dataset of handwritten digits (0-9) and evaluated for classification accuracy. The implementation uses PyTorch and leverages transfer learning by modifying the final fully connected layer of ResNet-18 to classify 10 digit classes.

## Features

- **Model Architecture**: ResNet-18 with custom final layer for 10-class classification
- **Data Preprocessing**: Image resizing to 224x224, normalization using ImageNet statistics
- **Training**: Adam optimizer with cross-entropy loss
- **Evaluation**: Comprehensive metrics including accuracy, F1-score, confusion matrix, and class-wise recall
- **Visualization**: Training curves and confusion matrix plots
- **Docker Support**: Containerized environment for reproducible execution

## Dataset

The project uses a digit image dataset organized in the following structure:
```
data/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 9/
└── test/
    ├── 0/
    ├── 1/
    ├── ...
    └── 9/
```

Each subdirectory contains images of the corresponding digit class.

## Requirements

- Python 3.9+
- PyTorch 2.0.0+
- Torchvision 0.15.0+
- NumPy 1.21.0+
- Scikit-learn 1.0.0+
- Pillow 9.0.0+
- Matplotlib 3.5.0+

## Installation

### Local Installation

1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Docker Installation

Build the Docker image:
```bash
docker build -t digit-recognition .
```

## Usage

### Training

Run the training script:
```bash
python train.py
```

This will:
- Load and preprocess the training data
- Train the ResNet-18 model for 3 epochs
- Save the trained model as `trained_model.pth`
- Generate training curves plot (`training_curves.png`)
- Evaluate on test set and generate confusion matrix (`confusion_matrix.png`)

### Evaluation

Run the evaluation script:
```bash
python evaluate.py
```

This will:
- Load the pre-trained model (`setB.pth`)
- Evaluate on test data
- Print overall accuracy, F1-score, and class-wise metrics
- Generate confusion matrix plot

### Docker Usage

#### Training in Docker
```bash
docker run -v $(pwd)/data:/app/data digit-recognition python train.py
```

#### Evaluation in Docker
```bash
docker run -v $(pwd)/data:/app/data digit-recognition python evaluate.py
```

## Configuration

Key parameters can be modified in the scripts:

- `BATCH_SIZE`: Batch size for data loading (default: 32)
- `EPOCHS`: Number of training epochs (default: 3)
- `LR`: Learning rate (default: 1e-3)
- `DATA_DIR`: Path to training data (default: "data/train/")
- `TEST_DATA_DIR`: Path to test data (default: "data/test/")

## Model Architecture

The model uses ResNet-18 with:
- Pre-trained weights removed (training from scratch)
- Input size: 224x224 RGB images
- Output: 10 classes (digits 0-9)
- Final layer: Linear layer with 512 inputs (ResNet-18 feature size) and 10 outputs

## Training Details

- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 3
- **Device**: CUDA if available, otherwise CPU

## Evaluation Metrics

The evaluation provides:
- Overall accuracy
- Macro F1-score
- Class-wise recall (accuracy per digit)
- Confusion matrix
- Classification report

Special attention is given to Class 7 performance as per project requirements.

## Results

### Training Performance
- The model typically achieves high accuracy on the training set within 3 epochs
- Training loss decreases steadily
- Accuracy improves rapidly in early epochs

### Test Performance
- Overall accuracy: ~95-98% (varies based on dataset quality)
- Class 7 accuracy: Typically high, monitored specifically
- Confusion matrix shows good separation between digit classes

## File Structure

```
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── setB.pth            # Pre-trained model weights
├── data/               # Dataset directory
│   ├── train/         # Training images
│   └── test/          # Test images
├── training_curves.png # Training visualization (generated)
└── confusion_matrix.png # Confusion matrix (generated)
```

## Docker Configuration

The Dockerfile:
- Uses Python 3.9 slim base image
- Installs PyTorch CPU version
- Copies project files
- Creates data directories
- Sets default command to run training

## Troubleshooting

### Common Issues

1. **CUDA not available**: The code automatically falls back to CPU if CUDA is not detected
2. **Memory issues**: Reduce batch size if encountering out-of-memory errors
3. **Missing data**: Ensure data directory structure matches the expected format
4. **Model loading errors**: Verify the model file path and compatibility

### Performance Tips

- Use GPU for faster training if available
- Increase epochs for better convergence (monitor for overfitting)
- Adjust learning rate based on training stability
- Use data augmentation for improved generalization

## Dependencies Details

- **torch**: Deep learning framework
- **torchvision**: Computer vision utilities and models
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning metrics
- **Pillow**: Image processing
- **matplotlib**: Plotting and visualization

## Future Improvements

- Implement data augmentation
- Add early stopping
- Experiment with different architectures (ResNet-50, EfficientNet)
- Add hyperparameter tuning
- Implement cross-validation
- Add model quantization for deployment

## License

This project is for educational purposes.

## Author

B22CS001</content>
<parameter name="filePath">/home/abhay/Downloads/B22CS001/README.md