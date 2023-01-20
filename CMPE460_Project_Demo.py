import yfinance as yf
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import math

plt.style.use('fivethirtyeight')

st.title("Stock Market Volatility Predictor")

ticker = st.text_input("Enter the ticker(AMZN, BTC-USD,...): ")
start_date = st.text_input("Enter the start date(YYYY-MM-DD): ")
end_date = st.text_input("Enter the end date(YYYY-MM-DD): ")
tickers_list = [ticker]

if start_date > end_date:
    st.write("Invalid date range. Start date must be befor end date.")
else:
    stock_data = yf.download(tickers_list, threads= False, start=start_date, end=end_date)
    
stock_data = stock_data[['Close']]
stock_data_array = stock_data.values

stock_data.head()

stock_data['Close'].plot(figsize=(15,5), legend=True)
plt.legend('Close Price @Apple')
plt.title('Close Price History')
plt.xlabel('Date')
plt.ylabel('Close Price ($)')
plt.show()

st.line_chart(stock_data['Close'])
st.write("Close Price History")

# Scale the data using the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(np.array(stock_data).reshape(-1,1))

# 90 days of data is taught
day_to_use = 90
x_train, y_train = [], []
for i in range(day_to_use, len(scaled)):
    x_train.append(scaled[i-day_to_use:i])
    y_train.append(scaled[i])

# Set train_size and test_size
train_size = math.ceil(len(stock_data)*0.8)
test_size = len(scaled) - train_size

# Splitting data between train and test
sd_train, sd_test = scaled[0:train_size,:], scaled[train_size-day_to_use:, :]

# Reshaping data to fit into LSTM model
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units = 50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(units = 50))
model.add(Dense(units = 1))
model.summary()

#Training model with adam optimizer and mse
model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='adam')
# Fit the model
model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1)

# To create test values
x_test = []
y_test = stock_data_array[train_size:, :]
for i in range(day_to_use, len(sd_test)):
    x_test.append(sd_test[i-day_to_use:i])

x_test = np.array(x_test)

# Make predictions on the test set
prediction = model.predict(x_test)

# Scale the data back to its original form
prediction = scaler.inverse_transform(prediction)
y_test = scaler.inverse_transform(y_train)

# Calculate the model's accuracy
accuracy = model.evaluate(x_train, y_train, verbose = 0)
st.write("Accuracy: {:.4f}%".format(accuracy[1]))

# Shows the value of "Predictions"
train = stock_data[:train_size]
close_pred = stock_data[train_size:]
close_pred['Predictions'] = prediction

close_pred

# Plot to all data
plt.figure(figsize=(14,4))
plt.title('Stock Market Volatility Predictor')
plt.ylabel('Close Price ($)')
plt.plot(close_pred[['Close', 'Predictions']])
plt.legend(['Actual Values', 'Predictions'])
plt.show()

st.line_chart(close_pred[['Close', 'Predictions']])
st.write('Stock Market Volatility Predictor')

# Test predict will predict the next values
testPredict = model.predict(x_test)
testPredict = scaler.inverse_transform(testPredict)

# Predict the next 10 days
last_val = testPredict[-1]
last_val_scaled = last_val / last_val
next_val = model.predict(np.reshape(last_val_scaled, (1,1,1)))
next_val = next_val.flatten()
st.write("Next 10 Days stock price:", np.concatenate((last_val, next_val), axis=0))

# Created to predict the next day's stock market volatility
stock_latest = yf.Ticker(ticker).history(period='1d')
stock_latest = stock_latest[['Close']]

# Sclae the data using the MinMaxScaler
stock_latest = scaler.transform(stock_latest)
stock_latest = np.reshape(stock_latest, (stock_latest.shape[0], stock_latest.shape[1], 1))

# Predicts the next day's value
next_day_predict = model.predict(stock_latest)
next_day_predict = scaler.inverse_transform(next_day_predict)
st.write("Predicted close price for next day: ", next_day_predict[0][0])