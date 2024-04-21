# MNIST Classification Model

This repository contains a Python script that trains a neural network model to classify handwritten digits using the MNIST dataset. The model is built using TensorFlow and Keras.

## Table of Contents

- [MNIST Classification Model](#mnist-classification-model)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Model Architecture](#model-architecture)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Results](#results)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

## Introduction

The MNIST dataset is a widely-used benchmark dataset in the field of machine learning. It consists of 60,000 training images and 10,000 testing images of handwritten digits (0 through 9), each with a resolution of 28x28 pixels.

In this project, we train a neural network model to classify these handwritten digits based on the pixel values of the images.

## Dataset

The MNIST dataset is included with the TensorFlow library and can be easily loaded using the `mnist.load_data()` function. It is divided into training and testing sets, with corresponding labels indicating the digit represented by each image.

## Model Architecture

The neural network model used for this classification task consists of three dense layers with ReLU activation functions, followed by a softmax output layer for multi-class classification.

- Input Layer: 784 nodes (flattened 28x28 image)
- Hidden Layer 1: 100 nodes with ReLU activation
- Hidden Layer 2: 50 nodes with ReLU activation
- Hidden Layer 3: 25 nodes with ReLU activation
- Output Layer: 10 nodes with softmax activation

## Training

The model is trained using the Adam optimizer and sparse categorical crossentropy loss function. Training is performed for 5 epochs with a batch size of 32.

## Evaluation

After training, the model is evaluated on the test set to measure its performance in terms of loss and accuracy.

Additionally, we generate a confusion matrix to visualize the model's performance in classifying each digit.

## Results

The trained model achieves an accuracy of approximately [insert accuracy] on the test set.

The confusion matrix provides insights into the model's performance for each digit class, highlighting areas of strength and weakness.

## Usage

To use this code, follow these steps:

1. Clone the repository:

```
git clone https://github.com/Sai-004/MNIST-Digit-Classification-Model.git
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Run the Python script:

```
python model.py
python mnist_classification.py
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.