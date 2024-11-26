import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
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

# Extract the 'Close Price' column for model training
close_prices = data_last_10_years['Close Price'].values.reshape(-1, 1)

# Scale the close prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close_prices = scaler.fit_transform(close_prices)

# Prepare data sequences for training (e.g., sequences of 60 days)
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Train and predict function with model checkpointing and early stopping
def train_and_predict(sequence_length=60, epochs=100, batch_size=128, units=50, dropout_rate=0.2, threshold_mae=1.0):
    X_train, y_train = create_sequences(scaled_close_prices, sequence_length)
    
    # Reshape X_train for LSTM input [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Build the LSTM model using Input() instead of input_shape
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))  # Explicit input layer
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))  # Predicting the next closing price

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Initialize callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Change to `.keras` extension as required
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

    # Predict the closing price for the last 'sequence_length' days
    last_sequence = scaled_close_prices[-sequence_length:]
    last_sequence_scaled = last_sequence.reshape(1, -1, 1)

    # Predict the next closing price
    predicted_price_scaled = model.predict(last_sequence_scaled)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    # Calculate MAE and RMSE for the last prediction
    actual_price = close_prices[-1][0]  # Last actual closing price
    mae = mean_absolute_error([actual_price], predicted_price[0])
    rmse = np.sqrt(mean_squared_error([actual_price], predicted_price[0]))

    return mae, rmse, predicted_price[0][0], actual_price

# Hyperparameter optimization function
def optimize_hyperparameters(sequence_lengths, batch_sizes, epochs_list, dropout_rates, units_list, max_iterations=10, threshold_mae=1.0):
    best_mae = float('inf')
    best_params = {}
    results = []

    # Using tqdm for progress bar
    total_combinations = len(sequence_lengths) * len(batch_sizes) * len(epochs_list) * len(dropout_rates) * len(units_list)
    progress_bar = tqdm(total=total_combinations, desc="Hyperparameter Tuning")

    for sequence_length in sequence_lengths:
        for batch_size in batch_sizes:
            for epochs in epochs_list:
                for dropout_rate in dropout_rates:
                    for units in units_list:
                        mae, rmse, predicted, actual = train_and_predict(
                            sequence_length=sequence_length,
                            epochs=epochs,
                            batch_size=batch_size,
                            units=units,
                            dropout_rate=dropout_rate,
                            threshold_mae=threshold_mae
                        )
                        
                        # Log the results
                        logger.info(f"Sequence Length: {sequence_length}, Batch Size: {batch_size}, Epochs: {epochs}, Dropout: {dropout_rate}, Units: {units} --> MAE: {mae:.4f}, RMSE: {rmse:.4f}, Predicted: {predicted:.2f}, Actual: {actual:.2f}")
                        
                        # Save the result
                        results.append({
                            "sequence_length": sequence_length,
                            "batch_size": batch_size,
                            "epochs": epochs,
                            "dropout_rate": dropout_rate,
                            "units": units,
                            "mae": mae,
                            "rmse": rmse,
                            "predicted_price": predicted,
                            "actual_price": actual
                        })

                        # Update best parameters if current MAE is lower
                        if mae < best_mae:
                            best_mae = mae
                            best_params = {
                                "sequence_length": sequence_length,
                                "batch_size": batch_size,
                                "epochs": epochs,
                                "dropout_rate": dropout_rate,
                                "units": units
                            }
                            logger.info(f"*** New Best MAE: {best_mae:.4f} with params: {best_params} ***")
                        
                        progress_bar.update(1)
    
    progress_bar.close()
    
    # Log the optimal hyperparameters
    logger.info(f"Optimal Hyperparameters: {best_params} with MAE: {best_mae:.4f}")
    
    # Save all results to a CSV file for further analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv("all_hyperparameter_results.csv", index=False)
    logger.info("All hyperparameter tuning results saved to all_hyperparameter_results.csv")
    
    return best_params, best_mae, results

# Define hyperparameter ranges for optimization
sequence_lengths = [60, 120]
batch_sizes = [32, 64, 128]
epochs_list = [50, 100, 150]
dropout_rates = [0.2, 0.3]
units_list = [50, 100]

# Run the hyperparameter optimization
best_params, best_mae, all_results = optimize_hyperparameters(
    sequence_lengths,
    batch_sizes,
    epochs_list,
    dropout_rates,
    units_list,
    max_iterations=10,
    threshold_mae=1.0
)

# After execution, the results will be available in the log file and the CSV file.
