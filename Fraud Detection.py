import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
import joblib
import os
import argparse

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
VERSION = "credit_card_classifier"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(path):
    """Load credit card transactions CSV with 'Time', features V1...V28, 'Amount', and 'Class' label."""
    df = pd.read_csv(path)
    # Rename columns for consistency
    if 'Time' in df.columns:
        df.rename(columns={'Time': 'timestamp'}, inplace=True)
    return df


def preprocess(df):
    """Scale numeric features and encode categorical if present."""
    df = df.copy()
    # Ensure timestamp type if exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', origin='unix', errors='ignore')

    # Identify label and features
    if 'Class' not in df.columns:
        raise ValueError("Data must contain 'Class' column for fraud label.")

    labels = df['Class'].values
    features = df.drop(columns=['Class', 'timestamp'], errors='ignore')

    # Scale numeric features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, labels, scaler


def create_sequences(X, y, seq_len=SEQ_LEN):
    """
    Build sequences of shape (n_samples, seq_len, n_features) and corresponding labels (label at last step).
    """
    seq_X, seq_y = [], []
    for i in range(len(X) - seq_len + 1):
        seq_X.append(X[i:i + seq_len])
        seq_y.append(y[i + seq_len - 1])
    return np.array(seq_X), np.array(seq_y)


def build_classification_model(n_features, seq_len=SEQ_LEN):
    """LSTM-based classifier for credit card fraud."""
    inp = Input(shape=(seq_len, n_features))
    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate(model, X_train, y_train, X_val, y_val):
    """Train model with class weights and callbacks, then evaluate on validation set."""
    # Compute class weights to address imbalance
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(weights))

    # Callbacks
    ckpt_path = os.path.join(MODEL_DIR, f"best_{VERSION}.h5")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=False  # for temporal data
    )

    # Validation metrics
    y_pred_prob = model.predict(X_val).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_prob)
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall:    {recall:.4f}")
    print(f"Validation F1-score:  {f1:.4f}")
    print(f"Validation ROC AUC:    {roc_auc:.4f}")

    return model


def main(data_path):
    # Load and preprocess data
    df = load_data(data_path)
    X, y, scaler = preprocess(df)

    # Create sequences
    X_seq, y_seq = create_sequences(X, y)

    # Split by time: first 70% train, next 15% val, last 15% test
    n = len(X_seq)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val     = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test   = X_seq[val_end:], y_seq[val_end:]

    # Build and train classifier
    n_features = X_train.shape[2]
    model = build_classification_model(n_features)
    model = train_and_evaluate(model, X_train, y_train, X_val, y_val)

    # Evaluate on test set
    print("\n--- Test Set Performance ---")
    y_test_prob = model.predict(X_test).ravel()
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    print(f"Test Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"Test Recall:    {recall_score(y_test, y_test_pred):.4f}")
    print(f"Test F1-score:  {f1_score(y_test, y_test_pred):.4f}")
    print(f"Test ROC AUC:    {roc_auc_score(y_test, y_test_prob):.4f}")

    # Save model and scaler
    model.save(os.path.join(MODEL_DIR, f"final_{VERSION}.h5"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{VERSION}.save"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM Classifier for Credit Card Fraud Detection')
    parser.add_argument('--data', type=str, required=True, help='Path to credit card transactions CSV')
    args = parser.parse_args()
    main(args.data)
