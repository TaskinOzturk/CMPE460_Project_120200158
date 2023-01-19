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

x_train, y_train = [], []
for i in range(120, len(scaled)):
    x_train.append(scaled[i-120:i])
    y_train.append(scaled[i])

train_size = math.ceil(len(stock_data)*0.8)
test_size = len(scaled) - train_size

#Splitting data between train and test
sd_train, sd_test = scaled[0:train_size,:], scaled[train_size:len(scaled),:1]

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units = 50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(units = 50, return_sequences=True))
model.add(LSTM(units = 50))
model.add(Dense(units = 1))
model.summary()

# Compile and fit the model
model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='adam')
model.fit(x_train, y_train, epochs=25, batch_size=32)

test_data = scaled[train_size-120:, :]
x_test = []
y_test = stock_data_array[train_size:, :]
for i in range(120, len(test_data)):
    x_test.append(test_data[i-120:i])

x_test = np.array(x_test)

# Make predictions on the test set
prediction = model.predict(x_test)

# Scale the data back to its original form
prediction = scaler.inverse_transform(prediction)
y_test = scaler.inverse_transform(y_train)

# Calculate the model's accuracy
accuracy = model.evaluate(x_train, y_train, verbose = 0)
print("Accuracy: ", accuracy)

train = stock_data[:train_size]
close_pred = stock_data[train_size:]
close_pred['Predictions'] = prediction

plt.figure(figsize=(15,5))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close PRice ($)')
plt.plot(close_pred[['Close', 'Predictions']])
plt.legend(['Val', 'Preds'])
plt.show()