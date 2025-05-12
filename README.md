## Credit Card Fraud Detection using LSTM Classifier

This repository implements a supervised LSTM-based classifier specifically designed to detect credit card fraud from sequential transaction data.

## Features

Sequence-based classification: Uses sliding windows of transactions (default length: 30) to capture temporal patterns.

Bi-directional LSTM architecture: Two stacked bidirectional LSTM layers with dropout regularization and a dense output layer.

Imbalance handling: Automatically computes class weights to compensate for the rarity of fraud cases.

Comprehensive metrics: Reports precision, recall, F1‑score, and ROC AUC on both validation and test sets.

Model persistence: Saves both the best checkpoint and final model, along with the feature scaler, for reproducible inference.

## Requirements

Python 3.7 or higher

TensorFlow 2.x

scikit-learn

pandas

NumPy

joblib

## Install dependencies via:

pip install -r requirements.txt

## Installation

git clone https://github.com/JaCar-868/Fraud-Detection-using-LSTM-based-RNN.git
cd Fraud-Detection-using-LSTM-based-RNN

## Usage

Run the classifier with:

python fraud_detection_improved.py --data PATH_TO_CSV

## Arguments:

--data : Path to the CSV file containing credit card transactions. The file must include:

Time (or timestamp): Epoch seconds (will be converted internally).

Feature columns V1…V28 and Amount.

Class label: 0 = legitimate, 1 = fraud.

The script will:

Load and scale the transaction features.

Build overlapping sequences of length defined by SEQ_LEN in the script (default 30).

Split data chronologically into train (70%), validation (15%), and test (15%) sets.

Train the LSTM classifier with callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau).

Report validation and test metrics (precision, recall, F1, ROC AUC).

Save the following to models/:

best_credit_card_classifier.h5

final_credit_card_classifier.h5

scaler_credit_card_classifier.save

## Configuration

All hyperparameters live at the top of fraud_detection_improved.py:

SEQ_LEN: Window length for sequences.

BATCH_SIZE: Training batch size.

EPOCHS: Number of epochs to train.

LEARNING_RATE: Initial learning rate.

MODEL_DIR: Directory where models and scalers are saved.

Modify these values as needed before training.

## Deployment and Monitoring

Batch inference: Load the saved model and scaler in a separate script to score new CSVs.

Real-time inference: Integrate into streaming frameworks (e.g., Kafka, AWS Kinesis).

Drift detection: Regularly monitor performance metrics and retrain on fresh data as fraud patterns change.

## License

This project is released under the MIT License.

## Acknowledgments

Based on the [Kaggle Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/JaCar-868/Disease-Progression/blob/main/LICENSE) file for more details.
