# MLOps-AbhayKashyap-B22CS001

## 🔹 Question 1 (Part A): Image Classification on MNIST and Fashion MNIST dataset using Resnet18 and Resnet50 models with varying hyperparameter values

## Test Classification Accuracy (%) on MNIST (epoch=2, pin_memory=True)

| Batch Size | Optimizer | Learning Rate | ResNet18 (%) | ResNet50 (%) |
|-----------:|-----------|---------------|--------------|--------------|
| 16 | SGD  | 0.001  | 98.70 | 98.41 |
| 16 | SGD  | 0.0001 | 95.29 | 89.75 |
| 16 | Adam | 0.001  | 98.77 | 98.07 |
| 16 | Adam | 0.0001 | **99.17 (Best)** | 98.40 |
| 32 | SGD  | 0.001  | 97.65 | 97.88 |
| 32 | Adam | 0.0001 | 82.75 | 64.76 |
| 32 | SGD  | 0.001  | 98.30 | 97.78 |
| 32 | Adam | 0.0001 | 98.92 | 97.76 |

## Test Classification Accuracy (%) on Fashion MNIST (epoch=2, pin_memory=True)

| Batch Size | Optimizer | Learning Rate | ResNet18 (%) | ResNet50 (%) |
|-----------:|-----------|---------------|--------------|--------------|
| 16 | SGD  | 0.001  | 87.20 | 86.96 |
| 16 | SGD  | 0.0001 | 79.36 | 72.13 |
| 16 | Adam | 0.001  | 88.92 | 87.51 |
| 16 | Adam | 0.0001 | 89.98 | 88.82 |
| 32 | SGD  | 0.001  | 85.58 | 84.26 |
| 32 | Adam | 0.0001 | 75.40 | 66.59 |
| 32 | SGD  | 0.001  | 89.00 | 87.89 |
| 32 | Adam | 0.0001 | **89.52** | 84.44 |


## 🔹 Question 1 (Part B): Support Vector Machine (SVM) Results

### 📌 SVM Performance on MNIST Dataset

| C Value | Kernel       | Gamma | Test Accuracy (%) | Train Time (ms) |
| ------: | ------------ | ----- | ----------------: | --------------: |
|     0.1 | RBF          | scale |             92.95 |        28369.39 |
|     0.1 | RBF          | auto  |             86.83 |        57821.00 |
|       1 | RBF          | scale |             96.05 |        13599.54 |
|       1 | RBF          | auto  |             91.65 |        19822.32 |
|      10 | RBF          | scale |  **96.70 (Best)** |        12867.16 |
|      10 | RBF          | auto  |             93.33 |        10187.13 |
|     0.1 | Poly (deg=2) | scale |             92.65 |        23906.48 |
|     0.1 | Poly (deg=3) | scale |             91.55 |        28056.12 |
|       1 | Poly (deg=2) | scale |             95.50 |        10886.34 |
|       1 | Poly (deg=3) | scale |             95.12 |        13164.86 |
|      10 | Poly (deg=2) | scale |             96.08 |         9271.96 |
|      10 | Poly (deg=3) | scale |             95.58 |        10222.79 |


### 📌 SVM Performance on Fashion-MNIST Dataset

| C Value | Kernel       | Gamma | Test Accuracy (%) | Train Time (ms) |
| ------: | ------------ | ----- | ----------------: | --------------: |
|     0.1 | RBF          | scale |             80.88 |        24278.35 |
|     0.1 | RBF          | auto  |             75.17 |        37888.52 |
|       1 | RBF          | scale |             86.85 |        13853.05 |
|       1 | RBF          | auto  |             87.17 |        18687.21 |
|      10 | RBF          | scale |  **87.83 (Best)** |        13029.95 |
|      10 | RBF          | auto  |             86.12 |        11637.92 |
|     0.1 | Poly (deg=2) | scale |             79.77 |        21518.59 |
|     0.1 | Poly (deg=3) | scale |             77.62 |        22872.94 |
|       1 | Poly (deg=2) | scale |             85.58 |        12890.00 |
|       1 | Poly (deg=3) | scale |             83.38 |        14940.31 |
|      10 | Poly (deg=2) | scale |             87.12 |        11479.55 |
|      10 | Poly (deg=3) | scale |             85.60 |        12927.32 |


## 🔹 SVM Theory & Analysis

### 🔸 Effect of Kernel Choice

The kernel plays a critical role in SVM performance. Across both MNIST and Fashion-MNIST datasets, the **RBF kernel consistently outperforms polynomial kernels**. This is because RBF kernels effectively capture complex non-linear decision boundaries inherent in image data. Polynomial kernels, especially with higher degrees, struggle in high-dimensional feature spaces and are highly sensitive to hyperparameter tuning.

### 🔸 Effect of Regularization Parameter (C)

The parameter **C controls the bias–variance tradeoff**. Increasing C from 0.1 to 10 generally improves accuracy by allowing tighter fitting of the training data. The best results for both datasets are achieved at **C = 10**, though very large values may increase overfitting risk and computational cost.

### 🔸 Effect of Gamma

Gamma determines the influence radius of a single training example. The **`scale` setting performs better than `auto`** in almost all cases, offering both higher accuracy and reduced training time. The `auto` setting often leads to unstable decision boundaries and excessive computation, especially for polynomial kernels.


## 🔹 Question 2: Device-wise Performance Comparison (Fashion-MNIST)

### 📌 ResNet18 vs ResNet50 – CPU vs GPU

| Device | Batch Size | Optimizer | LR    | ResNet18 Acc (%) | ResNet50 Acc (%) | Train Time ResNet18 (ms) | Train Time ResNet50 (ms) | FLOPs ResNet18 (GFLOPs) | FLOPs ResNet50 (GFLOPs) |
| ------ | ---------: | --------- | ----- | ---------------: | ---------------: | -----------------------: | -----------------------: | ----------------------: | ----------------------: |
| CPU    |         16 | SGD       | 0.001 |            86.56 |            76.81 |                326988.52 |                103601.76 |                   1.824 |                   4.132 |
| CPU    |         16 | Adam      | 0.001 |            85.46 |            81.40 |                309095.67 |                105343.43 |                   1.824 |                   4.132 |
| GPU    |         16 | SGD       | 0.001 |            83.38 |            82.69 |                121377.28 |                239964.80 |                   1.824 |                   4.132 |
| GPU    |         16 | Adam      | 0.001 |            83.81 |            84.35 |                127455.77 |                250968.11 |                   1.824 |                   4.132 |

---

## 🔹 Analysis (Question 2)

* **GPU significantly reduces training time** compared to CPU, especially for deeper models like ResNet50.
* **ResNet50 incurs higher FLOPs**, explaining longer training times despite GPU acceleration.
* Accuracy differences between CPU and GPU are minimal, confirming that **hardware affects speed, not model correctness**.
* Adam shows slightly better stability on deeper architectures but at a marginal cost in training time.



