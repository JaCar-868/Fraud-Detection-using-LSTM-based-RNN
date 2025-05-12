import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import os

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration
SEQ_LEN = 30          # length of input sequences
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path):
    """
    Load transaction data from CSV. Expect a 'timestamp' and feature columns, and optional 'label'.
    """
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df.sort_values('timestamp', inplace=True)
    return df


def create_sequences(data, seq_len=SEQ_LEN):
    """
    Create overlapping sequences from data array.
    Returns array of shape (n_samples, seq_len, n_features).
    """
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)


def build_lstm_autoencoder(n_features, seq_len=SEQ_LEN):
    """
    Build an LSTM autoencoder for sequence reconstruction.
    """
    model = Sequential([
        Bidirectional(LSTM(64, activation='relu', return_sequences=True), input_shape=(seq_len, n_features)),
        Dropout(0.2),
        Bidirectional(LSTM(32, activation='relu', return_sequences=False)),
        RepeatVector(seq_len),
        Bidirectional(LSTM(32, activation='relu', return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(64, activation='relu', return_sequences=True)),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    return model


def train_model(model, x_train, x_val):
    """
    Train with callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
    """
    checkpoint_path = os.path.join(MODEL_DIR, 'best_autoencoder.h5')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    history = model.fit(
        x_train, x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, x_val),
        callbacks=callbacks,
        shuffle=True
    )
    return history


def detect_anomalies(model, x_data, threshold):
    """
    Compute reconstruction errors and flag anomalies above threshold.
    Returns boolean mask and errors.
    """
    reconstructions = model.predict(x_data)
    mse = np.mean(np.power(x_data - reconstructions, 2), axis=(1, 2))
    anomalies = mse > threshold
    return anomalies, mse


def main(data_path):
    # Load and preprocess
    df = load_data(data_path)
    features = df.drop(columns=['timestamp', 'label'], errors='ignore').values
    scaler = MinMaxScaler()

    # Split train/val/test by time (70/15/15)
    n = len(features)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_data = scaler.fit_transform(features[:train_end])
    val_data = scaler.transform(features[train_end:val_end])
    test_data = scaler.transform(features[val_end:])

    # Create sequences
    x_train = create_sequences(train_data)
    x_val = create_sequences(val_data)
    x_test = create_sequences(test_data)

    # Build and train model
    n_features = x_train.shape[2]
    autoencoder = build_lstm_autoencoder(n_features)
    train_model(autoencoder, x_train, x_val)

    # Determine threshold on validation set (e.g., 99th percentile)
    _, val_mse = detect_anomalies(autoencoder, x_val, threshold=float('inf'))
    threshold = np.percentile(val_mse, 99)
    print(f"Anomaly detection threshold: {threshold:.6f}")

    # Detect on test set
    anomalies, test_mse = detect_anomalies(autoencoder, x_test, threshold)

    # If labels present, evaluate
    if 'label' in df.columns:
        labels = df['label'].values[val_end + SEQ_LEN - 1:]
        precision, recall, f1, _ = precision_recall_fscore_support(labels, anomalies, average='binary')
        roc_auc = roc_auc_score(labels, test_mse)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

    # Save final model and scaler
    autoencoder.save(os.path.join(MODEL_DIR, 'final_autoencoder.h5'))
    import joblib
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.save'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LSTM Autoencoder for Fraud Detection')
    parser.add_argument('--data', type=str, required=True, help='Path to transactions CSV')
    args = parser.parse_args()
    main(args.data)
