# Project 12 - XOR Neural Network

## Description

This project demonstrates how a simple neural network can learn the XOR logic gate, a classic example that cannot be solved using a single linear model.

## Dataset

Input:

| X1 | X2 | Output |
| -- | -- | ------ |
| 0  | 0  | 0      |
| 0  | 1  | 1      |
| 1  | 0  | 1      |
| 1  | 1  | 0      |

## Model Architecture

* Input Layer: 2 neurons
* Hidden Layer: 4 neurons
* Output Layer: 1 neuron
* Activation Function: Sigmoid

## Loss Function

Binary Cross Entropy Loss (BCELoss)

## Optimizer

Stochastic Gradient Descent (SGD)

## Run Training

```bash
python train.py
```

## Run Prediction

```bash
python predict.py
```

## Output

The trained model is saved as:

```text
xor_model.pth
```
