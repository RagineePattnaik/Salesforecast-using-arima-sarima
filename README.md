# Salesforecast-using-arima-sarima
# ARIMA and Seasonal ARIMA Model for Time Series Forecasting

## Overview
This project demonstrates the implementation of time series forecasting using ARIMA (AutoRegressive Integrated Moving Average) and Seasonal ARIMA (SARIMA) models. The application is developed using Python and Streamlit to provide an interactive web interface that allows users to upload their own time series data, test for stationarity, and visualize the forecasting results.

## Key Features
- **File Upload**: Allows users to upload their own CSV files containing time series data.
- **Stationarity Testing**: Implements the Augmented Dickey-Fuller (ADF) test to check whether the data is stationary.
- **Differencing**: Differencing techniques are used to make the data stationary, including seasonal differencing for improved accuracy.
- **ARIMA and SARIMA Models**: Users can configure and apply both ARIMA and SARIMA models using adjustable parameters (p, d, q for ARIMA, P, D, Q, S for SARIMA).
- **Visualizations**: Dynamic plots to visualize historical sales, forecasts, autocorrelation, and partial autocorrelation.
- **Future Forecasting**: The app allows users to predict future values (12-24 months ahead) based on the historical data.

## Usage
This project is applicated as a web application using Pycharm platform with the help of Streamlit. First, we create a python project with virtual environment option and then inside our project, we and create a new python file by the name main1.py,  copy the respective code that is shown here in main1 python file. Then run in the terminal with command streamlit run main1.py. The project will be displayed in the local browser.
You can upload any files of csv format with Month and Sales column, where Month is the correct datetime format (e.g., yyyy-mm).
Do remember to install streamlit in your project file environment first.
