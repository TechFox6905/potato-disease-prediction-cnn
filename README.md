# 🥔 Potato Disease Prediction with CNN

This project uses a **Convolutional Neural Network (CNN)** model built with TensorFlow to classify potato plant leaf images into three categories:

- **Early Blight**
- **Late Blight**
- **Healthy**

It aims to support farmers and researchers in identifying diseases early using computer vision techniques.

---

## 🚀 Features

- Accurate image classification for three types of leaf conditions.
- End-to-end pipeline: from data preprocessing to model training and evaluation.
- Uses `EarlyStopping` and `ReduceLROnPlateau` for stable training.
- Visualizations of training and evaluation results.
- GPU-powered training (NVIDIA RTX 4050) for high performance.

---

## 📁 Project Structure

```bash
├── data/
│   ├── raw/               # Original image dataset
│   └── processed/         # Resized or augmented images
├── models/                # Saved model versions
├── notebooks/
│   └── potato_cnn.ipynb   # Experimentation and training notebook
├── outputs/               # Accuracy/loss curves, confusion matrices
├── src/
│   ├── data_loader.py     # Loads and prepares dataset
│   ├── evaluate.py        # Evaluates model on test data
│   ├── model.py           # CNN model architecture
│   └── train.py           # Training script
├── utils/
│   └── visualizer.py      # Visualization tools
└── requirements.txt       # Project dependencies
```
## 📊 Dataset Overview

- **Early Blight**: 1000 images  
- **Late Blight**: 1000 images  
- **Healthy**: 152 images  

⚠️ *Note: Dataset imbalance is mitigated using data augmentation.*

---

## 🧪 Preprocessing Steps

- **Image Resizing**: All images resized to `256x256`  
- **Channels**: 3 (RGB)  
- **Batch Size**: 32  
- **Augmentation**: Random flips, rotations, etc.  
- **Normalization**: Pixel values scaled from `0–255` to `0–1`  

---

## 🧠 Model Architecture

Custom CNN using `Sequential` API with ~184K trainable parameters:

| Layer            | Filters | Output Shape         |
|------------------|---------|----------------------|
| Conv2D + ReLU    | 32      | (254, 254, 32)       |
| MaxPooling2D     | -       | (127, 127, 32)       |
| Conv2D + ReLU    | 64      | (125, 125, 64)       |
| MaxPooling2D     | -       | (62, 62, 64)         |
| Conv2D + ReLU    | 64      | (60, 60, 64)         |
| MaxPooling2D     | -       | (30, 30, 64)         |
| Conv2D + ReLU    | 64      | (28, 28, 64)         |
| MaxPooling2D     | -       | (14, 14, 64)         |
| Conv2D + ReLU    | 64      | (12, 12, 64)         |
| ...              | ...     | ...                  |
| Dense + Softmax  | 3       | Class Probabilities  |

---

## ⚙️ Training Configuration

- **Epochs**: 20  
- **Image Size**: 256x256  
- **Batch Size**: 32  
- **Callbacks**:
  - `EarlyStopping(patience=3)`
  - `ReduceLROnPlateau(patience=2)`

🕒 **Training Time**: 2 min 32 sec  
💻 **Hardware Used**: NVIDIA RTX 4050 GPU

---

## 📈 Final Results

| Metric               | Value    |
|----------------------|----------|
| Train Accuracy       | 95.20%   |
| Validation Accuracy  | 86.98%   |
| Test Accuracy        | 92.97%   |
| Final Learning Rate  | 0.0010   |
| Final Loss (Test)    | 0.1481   |

```bash
Evaluating model...
8/8 [==============================] - 1s 23ms/step - loss: 0.1481 - accuracy: 0.9297
```

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/TechFox6905/potato-disease-prediction-cnn.git
cd potato-disease-prediction-cnn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🧪 How to Use

To train the model:

```bash
python src/train.py
```

To evaluate the trained model:

```bash
python src/evaluate.py
```
