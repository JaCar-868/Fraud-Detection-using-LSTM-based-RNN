# Fraud Detection using LSTM

## Overview

This repository contains an implementation of a Long Short-Term Memory (LSTM) network to detect anomalies in sequential transaction data. The goal is to identify potential fraud by detecting anomalies based on the reconstruction error of the LSTM model.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Fraud detection is a critical application of machine learning where the objective is to identify fraudulent transactions among a large volume of genuine transactions. This project demonstrates how LSTM networks can be used to detect such anomalies by learning the normal transaction patterns and identifying deviations.

## Installation

To run the code in this repository, you'll need to have Python installed along with the following libraries:
- NumPy
- Pandas
- scikit-learn
- TensorFlow

You can install the required libraries using pip:

pip install numpy pandas scikit-learn tensorflow

## Usage
1. Clone the repository:

git clone https://github.com/yourusername/fraud-detection-lstm.git
cd fraud-detection-lstm

2. Run the fraud_detection.py program:

python fraud_detection.py

### Code Explanation
## Import Libraries
The necessary libraries are imported, including NumPy for numerical operations, Pandas for data manipulation, scikit-learn for preprocessing, and TensorFlow for building the LSTM model.

## Data Generation
Sample data is generated for illustration purposes. Normal transaction data is created using a normal distribution, and anomalies are introduced by adding a fixed value to some of the data points.

## Data Preprocessing
The data is scaled using MinMaxScaler to bring all values into the range [0, 1]. The scaled data is then reshaped to be suitable for LSTM input, i.e., [samples, time steps, features].

## Train-Test Split
The data is split into training and testing sets. 80% of the data is used for training, and 20% is used for testing.

## LSTM Model
An LSTM model is built using TensorFlow's Keras API. The model consists of two LSTM layers followed by a dense layer. The model is compiled with the Adam optimizer and mean squared error (MSE) loss function.

## Model Training
The model is trained on the training data for 10 epochs with a batch size of 32. Validation is performed on the testing data.

## Anomaly Detection
The trained model is used to predict the transaction data. The mean squared error (MSE) between the actual and predicted values is calculated. A threshold is set to identify anomalies, and transactions with an MSE above this threshold are flagged as anomalies.

## Output
The number of detected anomalies is printed to the console.

## Contributing
Contributions are welcome! If you have any improvements or suggestions, please create a pull request or open an issue to discuss them.

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/JaCar-868/Disease-Progression/blob/main/LICENSE) file for more details.
