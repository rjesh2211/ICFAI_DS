#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:35:31 2025

@author: rajesh
"""

# ARIMA Model Implementation in Python
# ARIMA: AutoRegressive Integrated Moving Average

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load and prepare time series data
def load_data(file_path=None):
    """
    Load time series data from a file or create sample data
    
    Parameters:
    file_path (str): Path to the data file (optional)
    
    Returns:
    pd.Series: Time series data with datetime index
    """
    if file_path:
        # Load from file (CSV, Excel, etc.)
        # Adjust read function based on file type
        df = pd.read_csv(file_path, parse_dates=True, index_col=0)
        time_series = df.iloc[:, 0]  # Assuming first column contains the values
    else:
        # Create sample data if no file is provided
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        # Creating a sample time series with trend and seasonality
        trend = np.linspace(10, 30, 100)  # Upward trend
        seasonality = 5 * np.sin(np.linspace(0, 12*np.pi, 100))  # Seasonal pattern
        noise = np.random.normal(0, 1, 100)  # Random noise
        values = trend + seasonality + noise
        time_series = pd.Series(values, index=dates)
    
    print(f"Data shape: {time_series.shape}")
    return time_series

# Step 2: Check stationarity of the time series
def check_stationarity(time_series):
    """
    Check if time series is stationary using ADF test and visualizations
    
    Parameters:
    time_series (pd.Series): Input time series data
    
    Returns:
    bool: True if stationary, False otherwise
    """
    print("\n=== Stationarity Check ===")
    
    # Augmented Dickey-Fuller test
    adf_result = adfuller(time_series.dropna())
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    
    # Interpret the result
    is_stationary = adf_result[1] < 0.05
    print(f"Series is {'stationary' if is_stationary else 'non-stationary'} based on ADF test (p < 0.05)")
    
    # Visualize the time series
    plt.figure(figsize=(12, 8))
    
    # Plot the original time series
    plt.subplot(311)
    plt.plot(time_series)
    plt.title('Original Time Series')
    plt.xlabel('Date')
    plt.ylabel('Value')
    
    # Plot ACF (AutoCorrelation Function)
    plt.subplot(312)
    plot_acf(time_series.dropna(), ax=plt.gca(), lags=40)
    plt.title('Autocorrelation Function (ACF)')
    
    # Plot PACF (Partial AutoCorrelation Function)
    plt.subplot(313)
    plot_pacf(time_series.dropna(), ax=plt.gca(), lags=40)
    plt.title('Partial Autocorrelation Function (PACF)')
    
    plt.tight_layout()
    plt.show()
    
    return is_stationary

# Step 3: Make time series stationary if needed
def make_stationary(time_series, max_diff=2):
    """
    Transform a non-stationary time series to stationary through differencing
    
    Parameters:
    time_series (pd.Series): Input time series data
    max_diff (int): Maximum differencing order to attempt
    
    Returns:
    tuple: (differenced_series, d_value)
        - differenced_series (pd.Series): Stationary series after differencing
        - d_value (int): Order of differencing applied
    """
    diff_series = time_series.copy()
    d_value = 0
    
    # Apply differencing until stationary or max_diff is reached
    for d in range(1, max_diff + 1):
        diff_series = diff_series.diff().dropna()
        
        # Check if stationary after differencing
        adf_result = adfuller(diff_series.dropna())
        if adf_result[1] < 0.05:
            d_value = d
            print(f"\nSeries became stationary after {d} differencing")
            break
    
    if d_value == 0:
        print("\nSeries is already stationary, no differencing needed")
    elif d_value == max_diff and adf_result[1] >= 0.05:
        print(f"\nWarning: Series still non-stationary after {max_diff} differencing")
    
    # Visualize the differenced series
    plt.figure(figsize=(10, 6))
    plt.plot(diff_series)
    plt.title(f'Time Series After {d_value} Differencing')
    plt.xlabel('Date')
    plt.ylabel('Differenced Value')
    plt.show()
    
    return diff_series, d_value

# Step 4: Determine p, d, q parameters
def determine_pdq_parameters(stationary_series, d_value, max_p=5, max_q=5):
    """
    Analyze ACF and PACF plots to suggest p and q values
    
    Parameters:
    stationary_series (pd.Series): Stationary time series
    d_value (int): Order of differencing used
    max_p (int): Maximum AR order to consider
    max_q (int): Maximum MA order to consider
    
    Returns:
    tuple: Suggested (p, d, q) values
    """
    print("\n=== Parameter Determination ===")
    
    # Plot ACF and PACF to help determine p and q parameters
    plt.figure(figsize=(12, 6))
    
    plt.subplot(121)
    plot_acf(stationary_series.dropna(), ax=plt.gca(), lags=40)
    plt.title('ACF for Determining q')
    
    plt.subplot(122)
    plot_pacf(stationary_series.dropna(), ax=plt.gca(), lags=40)
    plt.title('PACF for Determining p')
    
    plt.tight_layout()
    plt.show()
    
    # Suggest p and q based on common interpretation rules
    # (This is a simplistic approach; in practice, you might want to try multiple values)
    print("\nParameter selection guidance:")
    print("1. p: Number of significant lags in PACF")
    print("2. d: Differencing order already determined as", d_value)
    print("3. q: Number of significant lags in ACF")
    
    p = int(input(f"Enter p value (suggested range: 0-{max_p}): "))
    q = int(input(f"Enter q value (suggested range: 0-{max_q}): "))
    
    return p, d_value, q

# Step 5: Fit ARIMA model
def fit_arima_model(time_series, p, d, q):
    """
    Fit ARIMA model with specified p, d, q parameters
    
    Parameters:
    time_series (pd.Series): Original time series data
    p (int): AR order
    d (int): Differencing order
    q (int): MA order
    
    Returns:
    model_fit: Fitted ARIMA model
    """
    print(f"\n=== Fitting ARIMA({p},{d},{q}) Model ===")
    
    # Create and fit the model
    model = ARIMA(time_series, order=(p, d, q))
    model_fit = model.fit()
    
    # Print model summary
    print(model_fit.summary())
    
    return model_fit

# Step 6: Forecast with the ARIMA model
def forecast_with_arima(model_fit, time_series, steps=30):
    """
    Generate and visualize forecasts using the fitted ARIMA model
    
    Parameters:
    model_fit: Fitted ARIMA model
    time_series (pd.Series): Original time series data
    steps (int): Number of steps to forecast
    
    Returns:
    pd.Series: Forecast values
    """
    print(f"\n=== Forecasting {steps} steps ahead ===")
    
    # Generate forecasts
    forecast_result = model_fit.get_forecast(steps=steps)
    forecast_mean = forecast_result.predicted_mean
    
    # Get confidence intervals
    conf_int = forecast_result.conf_int()
    
    # Create forecast dates
    last_date = time_series.index[-1]
    if isinstance(last_date, pd.Timestamp):
        freq = pd.infer_freq(time_series.index)
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq=freq)
    else:
        forecast_dates = np.arange(len(time_series), len(time_series) + steps)
    
    forecast_mean.index = forecast_dates
    conf_int.index = forecast_dates
    
    # Visualize the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(time_series, label='Observed')
    plt.plot(forecast_mean, color='red', label='Forecast')
    plt.fill_between(conf_int.index, 
                    conf_int.iloc[:, 0], 
                    conf_int.iloc[:, 1], 
                    color='pink', alpha=0.3)
    plt.title(f'ARIMA({model_fit.model.order[0]},{model_fit.model.order[1]},{model_fit.model.order[2]}) Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return forecast_mean

# Step 7: Evaluate model performance
def evaluate_model(model_fit, time_series, test_size=0.2):
    """
    Evaluate ARIMA model performance using train-test split
    
    Parameters:
    model_fit: Fitted ARIMA model
    time_series (pd.Series): Original time series data
    test_size (float): Proportion of data to use for testing
    
    Returns:
    dict: Dictionary of evaluation metrics
    """
    print("\n=== Model Evaluation ===")
    
    # Split data into train and test sets
    train_size = int(len(time_series) * (1 - test_size))
    train, test = time_series[:train_size], time_series[train_size:]
    
    # Fit model on training data
    p, d, q = model_fit.model.order
    train_model = ARIMA(train, order=(p, d, q)).fit()
    
    # Forecast for test period
    forecast_result = train_model.get_forecast(steps=len(test))
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()
    
    # Ensure forecast has the same index as test for comparison
    forecast_mean.index = test.index
    conf_int.index = test.index
    
    # Calculate error metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    mse = mean_squared_error(test, forecast_mean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test, forecast_mean)
    mape = np.mean(np.abs((test - forecast_mean) / test)) * 100
    
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Visualize actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(train, label='Training Data')
    plt.plot(test, label='Actual Test Data')
    plt.plot(forecast_mean, color='red', label='Forecast')
    plt.fill_between(conf_int.index, 
                    conf_int.iloc[:, 0], 
                    conf_int.iloc[:, 1], 
                    color='pink', alpha=0.3)
    plt.title('ARIMA Model Evaluation - Test Set Prediction')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }

# Step 8: Main function to run the full ARIMA workflow
def main():
    """
    Main function to execute the full ARIMA modeling workflow
    """
    print("=== ARIMA Time Series Analysis and Forecasting ===\n")
    
    # 1. Load data
    time_series = load_data()  # Use load_data('your_file.csv') to load your own data
    
    # 2. Check stationarity
    is_stationary = check_stationarity(time_series)
    
    # 3. Make stationary if needed
    if not is_stationary:
        stationary_series, d_value = make_stationary(time_series)
    else:
        stationary_series, d_value = time_series, 0
    
    # 4. Determine p, d, q parameters
    p, d, q = determine_pdq_parameters(stationary_series, d_value)
    
    # 5. Fit ARIMA model
    model_fit = fit_arima_model(time_series, p, d, q)
    
    # 6. Generate forecasts
    forecast = forecast_with_arima(model_fit, time_series)
    
    # 7. Evaluate model performance
    metrics = evaluate_model(model_fit, time_series)
    
    print("\n=== ARIMA Analysis Complete ===")

# Execute the script if run directly
if __name__ == "__main__":
    main()