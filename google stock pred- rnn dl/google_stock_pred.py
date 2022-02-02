'''from deep learning A-Z course
   coded by trishit nath thakur'''


# Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing training set


dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values   # we will be using only the open column for prediction however one can use other col too


# Feature Scaling


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))  # Transform features by scaling each feature to a given range here it is between 0 and 1
training_set_scaled = sc.fit_transform(training_set)


# Creating a data structure with 60 timesteps and 1 output


X_train = [] # list containing all rows for prediction purpose 
y_train = [] # list containing real values

for i in range(60, 1258):

    X_train.append(training_set_scaled[i-60:i, 0])   # we will use last 60 row values to predict the next outcome
    y_train.append(training_set_scaled[i, 0])  # y_train will contain real value

X_train, y_train = np.array(X_train), np.array(y_train)  # change list to numpy array so that they can be used in neural network


# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # we create new dimensionality so that if we want to add other indicators we can add them 
  
    # 3D tensor with shape(batch_size, timestamps, input_dims)
    # so we changed from 2D np array to 3D with the last corresponding to the indicator 

# Importing Keras libraries and packages


from keras.models import Sequential # to add sequence of layers
from keras.layers import Dense # to add output layer
from keras.layers import LSTM # to add LSTM layer
from keras.layers import Dropout  # to prevent overfitting


regressor = Sequential() # initialize our Recurrent Neural Network, Regressor is now an object of the sequential class

# Adding the first LSTM layer 
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1))) # adding our first LSTM layer.

# units is number of LSTM cells or units
# return_sequences = True as we are building stacked LSTM with more layers to come
# input_shape is shape of the input containing x_train corresponding to the time steps, and the indicators. batch_size taken automatically.

regressor.add(Dropout(0.2)) # the rate of neurons you wanna drop in the layers to do this regularisation and to drop 20% of your neurons in the layer.

# Adding a second LSTM layer 
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer 
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer 
regressor.add(LSTM(units = 50)) # the default value of the return sequence's parameter is false
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1)) # a classic fully connected layer using dense class

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') # rms_prop is recommended by keras for RNN but adam is good enough

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Making the predictions and visualising the results

# real stock price of 2017


dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# predicted stock price of 2017


dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0) # axis is axis along which you want to concatenate, here we used vertical axis(0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values # for first day of 2017 we need last 60 days data ie lower bound first day'17 - 60
inputs = inputs.reshape(-1,1) # to get numpy array with observations in lines and one or several columns
inputs = sc.transform(inputs)

X_test = []

for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test) 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # 3D tensor with shape(batch_size, timestamps, input_dims)

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # inverse transformation


# plotting results


plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')

plt.legend()
plt.show()