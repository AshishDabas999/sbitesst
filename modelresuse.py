import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("testing_logs.log"), logging.StreamHandler()]
)

logger = logging.getLogger()

# Load the trained model
best_model_path = "model_seq120_bs128_epochs150_units100.keras"  # Replace with your best model path
model = load_model(best_model_path)

# Load new test data
test_data = pd.read_csv('./500112.csv')  # Replace with the path to your test data CSV

# Preprocess the test data
# Convert 'Date' to datetime format and sort the data
test_data['Date'] = pd.to_datetime(test_data['Date'], format='%d-%B-%Y')
test_data = test_data.sort_values('Date')

# Extract the 'Close Price' and 'No.of Shares' columns
close_prices_test = test_data['Close Price'].values.reshape(-1, 1)
volumes_test = test_data['No.of Shares'].values.reshape(-1, 1)

# Scale the test data using the same scalers as training data
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaler_volume = MinMaxScaler(feature_range=(0, 1))

# Fit scalers to the test data and transform
scaled_close_prices_test = scaler_close.fit_transform(close_prices_test)
scaled_volumes_test = scaler_volume.fit_transform(volumes_test)

# Prepare data sequences for testing with the correct sequence length
sequence_length = 120  # Match this with your training sequence length
def create_sequences(data, volumes, sequence_length):
    X = []
    for i in range(sequence_length, len(data)):
        X.append(np.hstack([data[i-sequence_length:i, 0], volumes[i-sequence_length:i, 0]]))  # Combine close prices and volume
    return np.array(X)

# Create sequences for the test set
X_test = create_sequences(scaled_close_prices_test, scaled_volumes_test, sequence_length)

# Reshape X_test for LSTM input [samples, time steps, features]
X_test = X_test.reshape((X_test.shape[0], sequence_length, 2))  # Two features: close price and volume

# Predict closing prices for the test data
predicted_prices_scaled = model.predict(X_test)

# Inverse transform the predicted prices
predicted_prices = scaler_close.inverse_transform(predicted_prices_scaled)

# Calculate MAE and RMSE
actual_prices = close_prices_test[sequence_length:]  # Actual prices corresponding to predictions
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))

# Log the results
logger.info(f"Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")

# Output the predicted and actual prices for comparison
comparison_df = pd.DataFrame({
    'Actual Price': actual_prices.flatten(),
    'Predicted Price': predicted_prices.flatten()
})

# Save comparison results to a CSV file
comparison_df.to_csv("predictions_comparison.csv", index=False)
logger.info("Predictions comparison saved to predictions_comparison.csv")

# Print the last few predictions
print(comparison_df.tail())
