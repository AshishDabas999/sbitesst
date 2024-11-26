import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, Dropout, Input, Attention
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import random
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from tqdm import tqdm  # For progress bar

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Optional: Force TensorFlow to use CPU for deterministic behavior
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training_logs.log"),
        logging.StreamHandler()  # Optional: Also log to console
    ]
)

logger = logging.getLogger()

# Load the CSV file
data = pd.read_csv('./500112.csv')

# Convert 'Date' to datetime format and sort the data
data['Date'] = pd.to_datetime(data['Date'], format='%d-%B-%Y')
data = data.sort_values('Date')

# Filter the data for the last 10 years
cutoff_date = datetime.now() - pd.DateOffset(years=10)
data_last_10_years = data[data['Date'] >= cutoff_date]

# Extract the 'Close Price' and 'No.of Shares' columns for model training
close_prices = data_last_10_years['Close Price'].values.reshape(-1, 1)
volumes = data_last_10_years['No.of Shares'].values.reshape(-1, 1)  # Using 'No.of Shares' as Volume

# Scale the close prices and volumes
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaled_close_prices = scaler_close.fit_transform(close_prices)

scaler_volume = MinMaxScaler(feature_range=(0, 1))
scaled_volumes = scaler_volume.fit_transform(volumes)

# Prepare data sequences for training (e.g., sequences of 120 days)
def create_sequences(data, volumes, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(np.hstack([data[i-sequence_length:i, 0], volumes[i-sequence_length:i, 0]]))  # Combine close prices and volume
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Train and predict function with self-attention and model checkpointing
def train_and_predict(sequence_length=120, epochs=100, batch_size=128, units=50, dropout_rate=0.2):
    X_train, y_train = create_sequences(scaled_close_prices, scaled_volumes, sequence_length)
    
    # Reshape X_train for LSTM input [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], sequence_length, 2))  # Two features: close price and volume

    # LSTM model with sequential self-attention
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    lstm_out = LSTM(units=units, return_sequences=True)(input_layer)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    
    attention_layer = Attention()([lstm_out, lstm_out])
    
    lstm_out2 = LSTM(units=units, return_sequences=False)(attention_layer)
    lstm_out2 = Dropout(dropout_rate)(lstm_out2)
    
    output_layer = Dense(units=1)(lstm_out2)  # Predicting the next closing price
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Initialize callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    checkpoint_path = f"model_seq{sequence_length}_bs{batch_size}_epochs{epochs}_units{units}.keras"
    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=0)

    # Train the model with early stopping and a validation split
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        shuffle=False,
        verbose=0  # Suppress per-epoch output
    )

    # Load the best model
    model.load_weights(checkpoint_path)

    return model, scaler_close  # Return the model and scaler for future use
# Define hyperparameters
sequence_length = 60
epochs = 50
batch_size = 64
units = 100
dropout_rate = 0.3

# Train the model
model, scaler_close = train_and_predict(sequence_length, epochs, batch_size, units, dropout_rate)

# ----------------------------------------
# Testing Phase
# ----------------------------------------

# Load the test data
test_data = pd.read_csv('./500112.csv')  # Replace with the path to your test data CSV

# Preprocessing test data
test_data['Date'] = pd.to_datetime(test_data['Date'], format='%d-%B-%Y')
test_data = test_data.sort_values('Date')

# Extract the 'Close Price' and 'No.of Shares' columns for testing
close_prices_test = test_data['Close Price'].values.reshape(-1, 1)
volumes_test = test_data['No.of Shares'].values.reshape(-1, 1)  # Using 'No.of Shares' as Volume

# Scale the test close prices and volumes
scaled_close_prices_test = scaler_close.transform(close_prices_test)

scaler_volume_test = MinMaxScaler(feature_range=(0, 1))
scaled_volumes_test = scaler_volume_test.fit_transform(volumes_test)

# Create sequences for the test set
X_test, _ = create_sequences(scaled_close_prices_test, scaled_volumes_test, sequence_length)

# Reshape X_test for LSTM input [samples, time steps, features]
X_test = X_test.reshape((X_test.shape[0], sequence_length, 2))  # Two features: close price and volume

# Predict closing prices for the test data
predicted_prices_scaled = model.predict(X_test)

# Inverse transform the predicted prices
predicted_prices = scaler_close.inverse_transform(predicted_prices_scaled)

# Evaluate the model
actual_prices = close_prices_test[sequence_length:]  # Align actual prices with predictions
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Save predictions to a CSV file
results_df = pd.DataFrame({
    'Actual Price': actual_prices.flatten(),
    'Predicted Price': predicted_prices.flatten()
})
results_df.to_csv("predictions.csv", index=False)
logger.info("Predictions saved to predictions.csv")
