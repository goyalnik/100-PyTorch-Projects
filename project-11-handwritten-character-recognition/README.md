# Project 11 - Handwritten Character Recognition (EMNIST)

## Overview

This project uses a Convolutional Neural Network (CNN) to classify handwritten digits and letters from the EMNIST Balanced dataset.

## Dataset

* EMNIST Balanced
* 47 Classes
* Digits + Letters

## Model Architecture

* Conv2D (1 → 32)
* MaxPool
* Conv2D (32 → 64)
* MaxPool
* Fully Connected (3136 → 256)
* Output Layer (256 → 47)

## Requirements

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Prediction

```bash
python predict.py
```

## Output

The trained model is saved as:

```text
model.pth
```
