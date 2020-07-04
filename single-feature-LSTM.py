# Recurrent Neural Network - LSTM

# ======================== Part 1 - Data Preprocessing ========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling - Using Normlization since we are using sigmoid fn as activation
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# IMPORTANT: Creating a data structure with 60 timesteps and 1 output
# A wrong timestep could lead to overfitting or nonsensical output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping - adding new dimensions for additional feature/indicator
# 3D tensor with shape (nb_samples, timesteps, input_dim)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# ======================== Part 2 - Building the RNN ========================
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

# Initialising the RNN
regressor = Sequential()

# Four LSTM layer and some Dropout regularisation - to avoid overfitting
# Using 50 neurons to capture complex relationship
# return_sequence = True for layer stacking
# input_shape = (timesteps, num of indicators)
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# Dropout Regularization reduces overfitting
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the Output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
# regressor.fit(input, ground_truth, epochs, batch_size)
# when deciding epochs, look for convergence of loss function
# nn will be trained for every 32 stock prices instead of updating weights per stock price
# what do we choose 
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
 
# ============== Part 3 - Predictions and visualising the results ==============
# Getting the real stock price 
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

# Get the last 60 days before Jan + prices prior to last day of Jan
inputs_p = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# Make it the correct numpy shape
inputs = inputs_p.reshape(-1,1)

# using the same scaler object
inputs = sc.transform(inputs)

# 3D structure required by the RNN for prediction
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])   
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)

# Inverse scaling from normalization
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

#======================================================================
#here are different ways to improve the RNN model:

# Hyperparameter Tuning
#Getting more training data: 
#  we trained our model on the past 5 years of the Google Stock Price but it would be even better to train it on the past 10 years.
#Increasing the number of timesteps: 
#  the model remembered the stock prices from the 60 previous financial days to predict the stock price of the next day. That’s because we chose a number of 60 timesteps (3 months). You could try to increase the number of timesteps, by choosing for example 120 timesteps (6 months).
#Adding some other indicators: 
#  if you have the financial instinct that the stock price of some other companies might be correlated to the one of Google, you could add this other stock price as a new indicator in the training data.
#Adding more LSTM layers: 
#  we built a RNN with four LSTM layers but you could try with even more.
#Adding more neurons in the LSTM layers: 
#  we highlighted the fact that we needed a high number of neurones in the LSTM layers to respond better to the complexity of the problem and we chose to include 50 neurones in each of our 4 LSTM layers. You could try an architecture with even more neurones in each of the 4 (or more) LSTM layers.

#======================================================================