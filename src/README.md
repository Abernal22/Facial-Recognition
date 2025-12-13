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

- To evaluate the effectiveness of popular pretrained CNN architectures for FER.
- To apply stratified K-fold cross-validation to improve reliability of results.
- To explore transformer-based representations using Vision Transformers (ViT) combined with classical classifiers.
- To produce clean, reproducible metrics for use in reports and presentations.
- To support multiple datasets under a unified 7-class emotion label space.

---

## Datasets

### Supported Datasets

This project supports the following facial emotion datasets:

- **FER2013**
- **CK+**
- (Optional) RAF-DB

Each dataset is treated as a multi-class classification problem with seven emotion categories.

### Dataset Organization

Datasets are expected to be organized into class-specific directories, where each directory corresponds to a facial emotion (e.g., anger, happiness, sadness).

Large datasets and preprocessed feature files are intentionally kept **outside of version control** and stored locally.

### Label Mapping

Different datasets may use different naming conventions for emotion classes. The training pipeline supports a configurable label mapping mechanism that maps dataset-specific labels into a consistent 7-class format. This ensures fair and consistent comparisons across datasets.

---

## Models Implemented

### CNN Transfer Learning Models

The following pretrained CNN architectures are supported via `tf.keras.applications`:

- ResNet50
- EfficientNetB0
- MobileNetV3Small
- VGG16
- InceptionV3

Each model uses ImageNet pretrained weights and is extended with a custom classification head consisting of fully connected layers and dropout. Depending on dataset size, the base network may be frozen or partially unfrozen.

### Transformer-Based Models

Transformer experiments are implemented using a PyTorch pipeline and include:

- Vision Transformer (ViT) feature extraction
- Support Vector Machine (SVM) classification on ViT embeddings
- Probability prediction utilities for evaluation

---

## Training Pipeline

### Cross-Validation

CNN models are trained using **Stratified K-Fold cross-validation**, which preserves class balance across folds. This approach is particularly important for smaller datasets such as CK+.

### Dynamic Regularization

Regularization strategies are adjusted automatically based on dataset size. Smaller datasets apply stronger regularization and freeze base layers, while larger datasets allow more fine-tuning of pretrained weights.

### Model Selection and Checkpointing

During training:

- Early stopping is used to prevent overfitting.
- The best model per fold is saved based on validation loss.
- The globally best fold is selected based on validation accuracy.

---

## How to Run the Code

### Environment Setup

Install dependencies using:

```bash
pip install -r requirements.txt
