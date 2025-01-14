#code for streamlit web application
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.tseries.offsets import DateOffset


# Function to test stationarity
def adfuller_test(sales):
    result = adfuller(sales)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    st.write("ADF Test Results")
    for value, label in zip(result, labels):
        st.write(f"{label}: {value}")
    if result[1] <= 0.05:
        st.write("Strong evidence against the null hypothesis(Ho), data is stationary.")
    else:
        st.write("Weak evidence against null hypothesis, time series is non-stationary.")


# Function to create and plot ARIMA Model
def create_arima_model(df, p, d, q):
    model = ARIMA(df['Sales'], order=(p, d, q))
    model_fit = model.fit()
    st.write(model_fit.summary())
    df['forecast'] = model_fit.predict(start=90, end=103, dynamic=True)
    df[['Sales', 'forecast']].plot(figsize=(12, 8))
    plt.title("ARIMA Model Forecast")
    st.pyplot()


# Function to create and plot Seasonal ARIMA Model
def create_sarima_model(df, p, d, q, P, D, Q, S):
    model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(p, d, q), seasonal_order=(P, D, Q, S))
    results = model.fit()
    st.write(results.summary())
    df['forecast'] = results.predict(start=90, end=103, dynamic=True)
    df[['Sales', 'forecast']].plot(figsize=(12, 8))
    plt.title("Seasonal ARIMA Forecast")
    st.pyplot()


# Upload CSV file function
def load_uploaded_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = ["Month", "Sales"]  # Ensure columns are named correctly
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    return df


# Sidebar for file upload
st.title('ARIMA and Seasonal ARIMA Model for Time Series Forecasting')

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is not None:
    df = load_uploaded_data(uploaded_file)
    st.write(df.head())  # Display the first few rows of the uploaded data

    # Visualization
    st.subheader('Time Series Plot')
    df.plot()
    plt.title('Sales Over Time')
    st.pyplot()

    # Test for Stationarity
    st.subheader("Test for Stationarity (ADF Test)")
    adfuller_test(df['Sales'])

    # Differencing to make the data stationary
    df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)
    df['Seasonal First Difference'] = df['Sales'] - df['Sales'].shift(12)
    df.dropna(inplace=True)

    # Display ACF and PACF plots
    st.subheader("Autocorrelation and Partial Autocorrelation")
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    plot_acf(df['Seasonal First Difference'], lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    plot_pacf(df['Seasonal First Difference'], lags=40, ax=ax2)
    st.pyplot()

    # ARIMA Model Parameters
    st.sidebar.subheader("ARIMA Model Parameters")
    p = st.sidebar.slider('p (AR Order)', 0, 5, 1)
    d = st.sidebar.slider('d (Differencing Order)', 0, 2, 1)
    q = st.sidebar.slider('q (MA Order)', 0, 5, 1)

    # Create ARIMA model button
    if st.sidebar.button("Create ARIMA Model"):
        create_arima_model(df, p, d, q)

    # Seasonal ARIMA Model Parameters
    st.sidebar.subheader("Seasonal ARIMA Model Parameters")
    P = st.sidebar.slider('P (Seasonal AR Order)', 0, 5, 1)
    D = st.sidebar.slider('D (Seasonal Differencing Order)', 0, 2, 1)
    Q = st.sidebar.slider('Q (Seasonal MA Order)', 0, 5, 1)
    S = st.sidebar.slider('S (Seasonal Period)', 12, 24, 12)

    # Create Seasonal ARIMA model button
    if st.sidebar.button("Create Seasonal ARIMA Model"):
        create_sarima_model(df, p, d, q, P, D, Q, S)

    # Future Predictions
    st.subheader("Forecasting Future Sales")
    future_dates = [df.index[-1] + DateOffset(months=x) for x in range(0, 24)]
    future_dates_df = pd.DataFrame(index=future_dates[1:], columns=df.columns)

    # Create forecast for future months
    if st.sidebar.button("Forecast Future Sales"):
        model = sm.tsa.statespace.SARIMAX(df['Sales'], order=(p, d, q), seasonal_order=(P, D, Q, S))
        results = model.fit()
        future_df = pd.concat([df, future_dates_df])
        future_df['forecast'] = results.predict(start=104, end=120, dynamic=True)
        future_df[['Sales', 'forecast']].plot(figsize=(12, 8))
        plt.title("Forecast for Future Sales")
        st.pyplot()

else:
    st.write("Please upload a CSV file to begin.")
