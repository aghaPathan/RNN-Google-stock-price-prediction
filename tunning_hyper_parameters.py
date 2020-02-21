
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
timesteps = 120
X_train = []
y_train = []
for i in range(timesteps, 1258):
    X_train.append(training_set_scaled[i-timesteps:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def brain_rnn(lstm_layers, drop_out_ratio, optimizer):
    # Initialising the RNN
    regressor = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = lstm_layers, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(drop_out_ratio))

    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = lstm_layers, return_sequences = True))
    regressor.add(Dropout(drop_out_ratio))

    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = lstm_layers, return_sequences = True))
    regressor.add(Dropout(drop_out_ratio))

    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = lstm_layers))
    regressor.add(Dropout(drop_out_ratio))

    # Adding the output layer
    regressor.add(Dense(units = 1))

    # Compiling the RNN
    regressor.compile(optimizer = optimizer, loss = 'mean_squared_error')
    
    return regressor


classifier = KerasClassifier(build_fn= brain_rnn)
parameters = {'batch_size': [32, 42],
              'nb_epoch': [100, 120],
              'lstm_layers': [66, 76],
              'drop_out_ratio': [0.2, 0.3],
              'optimizer': ['adam', 'rmsprop'] }
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring= 'neg_mean_squared_error')

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_score = grid_search.best_score_

print(best_parameters, best_score)

