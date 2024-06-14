import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Sample data generation (for illustration purposes)
np.random.seed(42)
normal_data = np.random.normal(0, 1, (1000, 10))
anomaly_data = np.random.normal(0, 1, (50, 10)) + 3  # adding anomalies
data = np.concatenate([normal_data, anomaly_data])

# Data preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Reshape data to [samples, time steps, features]
data_scaled = data_scaled.reshape((data_scaled.shape[0], 1, data_scaled.shape[1]))

# Train-test split
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(train_data.shape[1], train_data.shape[2]), return_sequences=True))
model.add(LSTM(50))
model.add(Dense(train_data.shape[2]))

model.compile(optimizer='adam', loss='mse')
model.fit(train_data, train_data, epochs=10, batch_size=32, validation_data=(test_data, test_data), verbose=2)

# Predict and detect anomalies
predicted = model.predict(test_data)
mse = np.mean(np.power(test_data - predicted, 2), axis=1)
threshold = np.max(mse)  # or use a statistical method to define the threshold

anomalies = mse > threshold
print(f"Anomalies detected: {np.sum(anomalies)}")