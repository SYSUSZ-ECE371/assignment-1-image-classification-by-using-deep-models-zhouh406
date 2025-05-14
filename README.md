[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/VkJVVOAn)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=19516896&assignment_repo_type=AssignmentRepo)
# Image Classification with Deep Learning Models

This is my personal project repository where I explore image classification using deep learning models with [MMPretrain](https://github.com/open-mmlab/mmpretrain). The goal of this project is to get hands-on experience with training and fine-tuning modern classification models on custom datasets. It includes two exercises that reflect different aspects of working with deep learning pipelines — from configuration-based training to writing custom scripts and generating reports.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Exercise 1](#exercise-1)
- [Exercise 2](#exercise-2)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Environment](#environment)
- [Contributing](#contributing)

## Introduction

In this project, I focus on applying deep learning techniques for image classification tasks using the MMPretrain framework. My aim is to deepen my understanding of model configuration, training workflows, and evaluation metrics by working through practical examples.

The project is divided into two parts:
- One focuses on training a model using a configuration file.
- The other involves writing a complete training script from scratch and documenting the results.

## Project Structure

Here’s how the files are organized:
```text
image-classification-mmpretrain/
├── Ex1/
│   ├── flower_dataset.zip
│   ├── config.py
│   └── trained_model.pth
└── Ex2/
    ├── report.pdf
    ├── main.py  
    └── work_dir/
        └── trained_model.pth
```

## Exercise 1

### Overview

I used a flower dataset (`flower_dataset.zip`) to train a classification model using a configuration-based approach. This involved setting up a `config.py` file that defines the model architecture, data pipeline, optimizer, and training settings.

### Key Files

- `config.py`: Contains all the configurations needed for training.
- `trained_model.pth`: The final saved model after training completes.
- `flower_dataset.zip`: A labeled dataset of flower images used for training and validation.

## Exercise 2

### Overview

In this part, I built a standalone Python script (`main.py`) to train a classification model without relying on external config files. I also generated a PDF report summarizing the training process, results, and insights gained.

### Key Files

- `main.py`: A complete training script that handles data loading, model definition, training loop, and evaluation.
- `report.pdf`: A summary of the experiments, including accuracy curves, confusion matrix, and key observations.
- `work_dir/`: Directory containing the automatically saved model checkpoint (`trained_model.pth`).

## Getting Started

If you'd like to run or extend this project locally, here’s what you’ll need.

### Prerequisites

Before installing, make sure you have the following tools installed:
- Python 3.10
- Git
- CUDA-capable GPU (for accelerated training)
- Basic knowledge of PyTorch and MMPretrain

### Installation
Environment
The development environment I used for this project is as follows:

```text
Python Version: 3.10
CUDA Version: 12.1
PyTorch Version: 2.3
NumPy Version: 1.26.0
MMCV Version: 2.2.0
Operating System: Linux / Windows / macOS (with CUDA support)
Using these versions ensures compatibility across components, especially between MMPretrain, MMCV, and PyTorch.
```

### Contributing
If you'd like to contribute to this project, feel free to open an issue or submit a pull request. I welcome improvements to the code, documentation, and training strategies.
