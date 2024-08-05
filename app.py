import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trend Analyze')

# Allow the user to upload a CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# Read data based on user input or uploaded file
if uploaded_file is not None:
    # Load data from the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Data loaded successfully.")
else:
    # Load historical stock data
    df = pd.read_csv('your_file.csv')
    st.write("Data loaded successfully.")

# Continue with the rest of the code...

# Display data summary
st.subheader('Data Summary')
st.write(df.describe())

# Visualization - Closing Price vs Time Chart
st.subheader('Closing Price vs Time Chart')
fig1 = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig1)

# For 100 days Moving average
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
plt.plot(ma100, 'b', label='Original Price')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig2)

# For 200 days moving average
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig3)

# Split data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load pre-trained model
model = load_model('my_model.keras')

# Testing
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, pd.DataFrame([data_testing.iloc[0].to_dict()])], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_prediction = model.predict(x_test)

scaler = scaler.scale_
scaler_factor = 1 / scaler[0]
y_prediction = y_prediction * scaler_factor
y_test = y_test * scaler_factor

# Final graph - Prediction vs Original
st.subheader('Prediction vs Original')
fig4 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_prediction, 'y', label='Prediction Price')
ma100 = df.Close.rolling(100).mean()
plt.plot(ma100)
plt.plot(df.Close)

plt.xlabel('Time')
plt.ylabel('Price')

plt.legend()
st.pyplot(fig4)
