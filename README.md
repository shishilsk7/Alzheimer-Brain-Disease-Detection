---
title: BrzinDiseaseDetection
emoji: 🌍
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
# Alzheimer's Disease Detection from MRI Scans

This application uses a deep learning model to classify Alzheimer's disease stages from brain MRI scans. The model is trained on MRI images labeled with 4 categories of Alzheimer's disease severity:

- **Mild Demented**: Early signs of Alzheimer's, with subtle memory problems.
- **Moderate Demented**: Noticeable cognitive decline and daily life disruptions.
- **Non Demented**: Healthy, no signs of Alzheimer's.
- **Very Mild Demented**: Early cognitive decline, possibly just noticeable memory loss.

## Overview

This Gradio app uses a pre-trained convolutional neural network (CNN) to predict the stage of Alzheimer's disease based on MRI scans. The app classifies input MRI images into one of the four categories listed above.

## How to Use

1. **Upload MRI Image**: You can upload a T1-weighted brain MRI image.
2. **Model Prediction**: The model will predict the Alzheimer's disease stage and display the result.

You can also interact with a sample image to see the model’s prediction without uploading anything.

## Model

The model is a convolutional neural network (CNN) trained on a dataset of brain MRI scans. The network includes two convolutional layers followed by fully connected layers that output a classification result for each input image.

### Model Architecture:
- **Conv1**: 3x3 convolution, 16 filters, ReLU activation.
- **Conv2**: 3x3 convolution, 32 filters, ReLU activation.
- **Fully Connected Layer 1**: Linear layer with 128 neurons, ReLU activation.
- **Fully Connected Layer 2**: Output layer with 4 neurons for the 4 classes.

## Dependencies

The following libraries are required to run the application:

- `torch` (for the PyTorch deep learning framework)
- `torchvision` (for image transformations and pretrained models)
- `gradio` (for creating the web interface)
- `PIL` (for image handling)

## Installation and Setup

To run the application locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://huggingface.co/spaces/your-space-name
   cd your-space-name

