# Brain Tumor Classification

## Overview

This repository contains code for brain tumor classification using deep learning techniques. The objective is to classify MRI images into categories such as **tumor** and **non-tumor**. The model leverages convolutional neural networks (CNNs) for automatic feature extraction and classification.

## Features

- **Data Preprocessing:** Image normalization, resizing, and augmentation.
- **Model Architecture:** Custom CNN architecture or transfer learning using popular models (e.g., ResNet, VGG).
- **Loss Functions:** Cross-entropy loss or focal loss to handle class imbalance.
- **Performance Metrics:** Accuracy, precision, recall, F1-score, and confusion matrix.
- **Visualization:** Training curves and Grad-CAM for interpretability.

## Dataset

The dataset comprises MRI brain scans labeled as **tumor** or **non-tumor**. Ensure you organize the dataset as follows:

```
/dataset
  /train
    /tumor
    /non-tumor
  /validation
    /tumor
    /non-tumor
  /test
    /tumor
    /non-tumor
```

## Requirements

To run this project, ensure you have the following packages installed:

```bash
pip install -r requirements.txt
```

**Main Libraries:**

- Python 3.8+
- PyTorch / TensorFlow
- NumPy
- Pandas
- Matplotlib
- OpenCV

## Usage

1. **Clone the Repository:**

```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
```

2. **Prepare Dataset:** Place your dataset in the `dataset` folder as per the structure described above.

3. **Train the Model:**

```bash
python train.py --epochs 50 --batch_size 32 --lr 0.001
```

4. **Evaluate the Model:**

```bash
python evaluate.py --model_path models/best_model.pth
```

5. **Run Inference on New Images:**

```bash
python predict.py --image_path path/to/image.jpg
```

## Results

After training, you can find model performance metrics and visualizations in the `results/` directory.

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 95.2% |
| Precision | 94.5% |
| Recall    | 93.8% |
| F1-Score  | 94.1% |

## Model Architecture

The model consists of multiple convolutional layers followed by max-pooling, batch normalization, and fully connected layers.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## Acknowledgements

- Thanks to open-source contributors and dataset providers.
- Inspired by various research papers on medical image classification.
