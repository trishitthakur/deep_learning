'''from machine learning A-Z
   coded by trishit nath thakur'''


# Importing the libraries


import numpy as np
import pandas as pd
import tensorflow as tf


# Part 1-Data Preprocessing


# Importing the dataset


dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:-1].values # first 3 columns are not helpful in prediction
y = dataset.iloc[:, -1].values # last col having exited or not information


# Encoding categorical data


# Label Encoding the "Gender" column


from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()

X[:, 2] = le.fit_transform(X[:, 2]) # label encoding gender col to 0 or 1 type values


# One Hot Encoding the "Geography" column


from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough') 

# remainder='passthrough' will mean that all columns not specified in the list of “transformers” will be passed through without transformation

X = np.array(ct.fit_transform(X)) 


# Splitting the dataset into the Training set and Test set


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

 # when random_state set to an integer, train_test_split will return same results for each execution.

# Feature Scaling


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()


X_train = sc.fit_transform(X_train)

  # fit_transform() is used on the training data so that we can scale the training data and also learn the scaling parameters of that data

X_test = sc.transform(X_test)

  # Using the transform method we can use the same mean and variance as it is calculated from our training data to transform our test data

# Part 2 - Building the ANN


# Initializing the ANN

ann = tf.keras.models.Sequential() #sequential class from models module of keras library which belong to tf. we have now a sequence of layers

# Adding the input layer and the first hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # unit which corresponds exactly to the number of neurons

# Adding the second hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # a fully connected neural network must be the rectifier activation function

# Adding the output layer

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # we want to predict a binary variable which can take the value one or zero

  # sigmoid act func will give you the probabilities that the binary outcome is one
  
# Part 3 - Training the ANN


# Compiling the ANN

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

  # adam optimizer that can perform stochastic gradient descent(update the weights in order to reduce the loss error between your predictions and the real results)

# Training the ANN on the Training set

ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

 # batch size parameter gives exactly the number of predictions you want to have in the batch to be compared to that same number of real results.

 #  neural network needs a certain amount of epochs in order to learn properly

# Part 4 - Making the predictions and evaluating the model 


# Predicting the result of a single observation


print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5) # making prediction for a speicific customer


# Predicting the Test set results

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5) 

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) #to represent prediction and real values side by side


# Making the Confusion Matrix


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
  
   '''A Confusion matrix is an N x N matrix used for evaluating the performance of a classification model, where N is the number of target classes
    The target variable has two values: Positive or Negative
     The columns represent the actual values of the target variable
     The rows represent the predicted values of the target variable'''


print(cm)
accuracy_score(y_test, y_pred)