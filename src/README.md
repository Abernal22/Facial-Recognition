# Facial Emotion Recognition (FER)

This repository implements a complete experimental pipeline for **Facial Emotion Recognition (FER)** using deep learning. The project focuses on evaluating **transfer-learning convolutional neural networks (CNNs)** and **transformer-based approaches** on standard facial emotion datasets, with an emphasis on reproducibility, cross-validation, and clean metric reporting suitable for academic papers and presentations.

This repository is intended for **research and coursework use**, not for production deployment.

---

## Overview

Facial Emotion Recognition is a computer vision problem that involves classifying facial expressions into discrete emotion categories such as anger, happiness, sadness, and surprise.

In this project, multiple pretrained CNN backbones and transformer-based models are evaluated under a unified experimental framework. The pipeline supports consistent training, validation, and comparison across datasets and architectures.

---

## Project Objectives

The objectives of this project are:

- To evaluate the effectiveness of popular pretrained CNN architectures for facial emotion recognition.
- To apply stratified K-fold cross-validation to improve the reliability of performance estimates.
- To explore transformer-based representations using Vision Transformers (ViT) combined with classical classifiers.
- To produce clean, reproducible metrics for use in reports and presentations.
- To support multiple datasets under a unified 7-class emotion label space.

---

## Datasets

### Supported Datasets

This project supports the following facial emotion datasets:

- **FER2013**
- **CK+**
- **RAF-DB**

Each dataset is treated as a multi-class classification problem with seven emotion categories.

### Dataset Organization

Datasets are expected to be organized into class-specific directories, where each directory corresponds to a facial emotion (e.g., anger, happiness, sadness).

Large datasets and preprocessed feature files are intentionally kept **outside of version control** and stored locally.

### Dataset Comparison

| Dataset | Number of Images | Environment | Expressions | Annotation Method |
|-------|------------------|-------------|-------------|-------------------|
| RAF-DB | ~29,672 | In-the-wild (Web) | 7 emotions | ~40 annotators per image, majority voting |
| FER2013 | ~35,887 | In-the-wild (Web) | 7 emotions | Auto-collected, refined via keyword queries and manual cleaning |
| CK+ | ~593 sequences (~981 images) | Lab-controlled | 7 emotions | Two certified FACS coders |

---

## Standardized Emotion Class Mapping

To enable fair cross-dataset evaluation, all datasets were mapped to a unified 7-class emotion label space.

| Standard Index | Emotion | CK+ Label | RAF-DB Label | FER2013 Label |
|---------------|---------|-----------|--------------|---------------|
| 0 | Anger | Anger | 6 | `angry` |
| 1 | Disgust | Disgust | 3 | `disgust` |
| 2 | Fear | Fear | 2 | `fear` |
| 3 | Happiness | Happiness | 4 | `happy` |
| 4 | Sadness | Sadness | 5 | `sad` |
| 5 | Surprise | Surprise | 1 | `surprise` |
| 6 | Neutral | Neutral | 7 | `neutral` |

---

## Models Implemented

### CNN Transfer Learning Models

The following pretrained CNN architectures are supported via `tf.keras.applications`:

- ResNet50  
- EfficientNetB0  
- MobileNetV3Small  
- VGG16  
- InceptionV3  

Each model uses ImageNet pretrained weights and is extended with a custom classification head consisting of fully connected layers and dropout. Depending on dataset size, the base network may be frozen or partially unfrozen during training.

### Transformer-Based Models

Transformer experiments are implemented using a PyTorch-based pipeline and include:

- Vision Transformer (ViT) feature extraction
- Support Vector Machine (SVM) classification on ViT embeddings
- Probability prediction utilities for evaluation

---

## Training Pipeline

### Cross-Validation

CNN models are trained using **Stratified K-Fold cross-validation**, which preserves class balance across folds. This approach is particularly important for smaller datasets such as CK+.

### Hyperparameter Selection Strategy

Hyperparameters are selected dynamically based on dataset size.

| Hyperparameter | Small Dataset (≤ 2000 samples) | Large Dataset (> 2000 samples) |
|---------------|-------------------------------|--------------------------------|
| Dense Layer Size | 128 | 256 |
| Dropout Rate | 0.7 | 0.7 |
| L2 Regularization | 0.01 | 0.01 |
| Learning Rate | 5 × 10⁻⁵ | 2 × 10⁻⁵ |
| Freeze Base Model | True | False |

### Model Selection and Checkpointing

During training:

- Early stopping is used to prevent overfitting.
- The best model per fold is saved based on validation loss.
- The globally best-performing fold is selected based on validation accuracy.

---

## Experiments Conducted

The following model–dataset combinations were trained and evaluated during this project.  
For reproducibility and repository size considerations, trained model artifacts are stored locally and are not included in this repository.

### FER2013
- ResNet50
- EfficientNetB0
- MobileNetV3Small
- VGG16
- InceptionV3

### CK+
- ResNet50
- EfficientNetB0
- MobileNetV3Small
- VGG16
- InceptionV3

### RAF-DB
- ResNet50
- VGG16
- InceptionV3

---

## Results Summary

### CNN Performance (5-Fold Cross-Validation)

| Model | Avg Train Acc (%) | Avg Val Acc (%) | Avg Val Loss | Best Epoch |
|------|-------------------|------------------|--------------|------------|
| VGG16 | 87.29 | 68.08 | 1.279 | 45 |
| MobileNetV3Small | 65.05 | 61.49 | 1.278 | 45 |
| EfficientNetB0 | 75.67 | 66.93 | 1.076 | 43 |
| ResNet50 | 82.42 | 67.85 | 1.058 | 45 |

### Validation and OOD Performance (AffectNet)

| Method | RAF-DB Acc | FER2013 Acc | CK+ Acc | OOD Acc | Macro-F1 |
|------|------------|-------------|---------|---------|----------|
| ResNet-18 | 67.9 | 65.0 | 64.0 | 55.2 | 52.8 |
| MobileNetV3Small | 61.5 | 54.5 | 63.0 | 49.6 | 46.9 |
| EfficientNetB0 | 66.9 | 64.8 | 66.0 | 57.1 | 54.6 |
| ViT-Tiny/16 | 68.5 | 66.2 | 65.4 | 58.3 | 55.9 |
| Swin-Tiny Transformer | 69.2 | 67.0 | 66.1 | 59.0 | 56.8 |
| CNN-only Ensemble | 70.6 | 68.1 | 67.4 | 60.8 | 58.6 |
| Transformer-only Ensemble | 71.3 | 69.0 | 68.2 | 62.1 | 60.2 |
| Hybrid Ensemble | **73.8** | **71.5** | **70.6** | **65.4** | **63.7** |

---

## How to Run the Code

### Environment Setup

Install dependencies using:

```bash
pip install -r requirements.txt
