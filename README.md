# Brain Tumor Classification

## Overview

This repository contains code for brain tumor classification using deep learning techniques. The objective is to classify MRI images into categories such as **Glioma**, **Meningioma**, etc. The model leverages convolutional neural networks (CNNs) for automatic feature extraction and classification.

## Features

- **Data Preprocessing:** Image normalization, resizing, and other augmentation.
- **Model Architecture:** Custom CNN architecture consisting of two CNN blocks followed by a fully-connected block. 
- **Loss Functions:** Cross-entropy loss or focal loss to handle class imbalance.
- **Performance Metrics:** Accuracy, precision, recall, F1-score, and confusion matrix.
- **Visualization:** Training curves and heatmap for interpretability.

## Dataset

The dataset comprises MRI brain scans labeled as **Glioma**, **Meningioma**, **Notumor**, **Pituitary**. Ensure you organize the dataset as follows:

```
/dataset
  /Testing
    /glioma
    /meningioma
    /notumor
    /pituitary
  /Training
    /glioma
    /meningioma
    /notumor
    /pituitary
```

## Requirements

To run this project, ensure you have the following packages installed:

```bash
pip install -r requirements.txt
```

**Main Libraries:**

- Python 3.9+
- PyTorch
- NumPy
- Pandas
- Matplotlib

## Usage

1. **Clone the Repository:**

```bash
git clone https://github.com/MohammadYs77/Brain-Tumor-Classification-with-PyTorch.git
```

2. **Prepare Dataset:** Place your dataset in the `brain_tumor_dataset` folder as per the structure described above.

3. **Train the Model:** Currently, there are four argumments written for the program and may increase in the future. The four arguments are the epoch, batch size, image resize, and the device on which the model should be trained (All arguments have default values written in the example script below).

```bash
python main.py --epochs 10 --batch 16 --resize 224 --device 0
```

## Results

After training, you can find model performance metrics and visualizations in the `Figs/` directory.

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 95.2% |
| Precision | 94.5% |
| Recall    | 93.8% |
| F1-Score  | 94.1% |

## Model Architecture

The model consists of multiple convolutional layers followed by max-pooling, batch normalization, and fully connected layers.

## Acknowledgements

- Thanks to open-source contributors and dataset providers.
- Inspired by various research papers on medical image classification.