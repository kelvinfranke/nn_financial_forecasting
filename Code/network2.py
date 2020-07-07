from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame as df
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from math import sqrt
import matplotlib
from matplotlib import pyplot
import numpy as np
from numpy import array
import random
import ReadingInput
from ReadingInput import ReadingInput
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
import matplotlib.pylab as plt
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import KFold, cross_val_score
from keras.layers import LeakyReLU
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K
import sys

reading_input = ReadingInput()
# Ik heb de functie uitgebreid zodat ie de max en de min van elke array onthoudt
data, max_arr, min_arr = reading_input.process_data()

X, y = list(), list()
for j in range(0, 146):
  sequence = data[j,:]
  X.append(sequence[:8])
  y.append(sequence[9:15])

# X, y, X_cross_val, y_cross_val = train_test_split(X, y, test_size = 0.33, random_state = 42)

X_val, y_val = list(), list()
for i in range (0, 146):
  sequence = data[i,:]
  X_val.append(sequence[7:15])
  y_val.append(sequence[-6:])
	
X = np.array(X)
y = np.array(y)

X_val = np.array(X_val)
y_val = np.array(y_val)

random_indices = np.arange(0, 143, 1).tolist()
np.random.shuffle(random_indices)

# voert k-fold cross validation uit
def kfold_procedure(X, y, model):
  k_fold = KFold(n_splits = 12)
  total_test = []
  save = 1
  best = -99999
  top_model = 0
  avg_acc = 0
  for train_indices, test_indices in k_fold.split(X):
    # print(train_indices)
    X_matrix = []
    y_matrix = []
    for i in train_indices:
      X_matrix.append(X[i])
    for i in train_indices:
      y_matrix.append(y[i])
    X_matrix = np.array(X_matrix)
    y_matrix = np.array(y_matrix)
    # print(np.shape(X_matrix))
    # print(np.shape(y_matrix))
    history = model.fit(X_matrix, y_matrix, epochs=100, verbose=0, shuffle=False)
    X_test = []
    y_test = []
    for i in test_indices:
      X_test.append(X[i])
    for i in test_indices:
      y_test.append(y[i])
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    results = model.evaluate(X_test, y_test, verbose=0)
    total_test.append(results)
    if results[1] > best:
      best = results[1]
      top_model = save
    avg_acc += results[1]
    # model.save('model' + str(save))
    save += 1
    # print(train_indices, test_indices)
  # print(total_test)
  print("Best model index: " + str(top_model))
  return avg_acc/12


# Dit stukkie veranderd de hoeveelheid neurons in het model en het gebruikt die kfold_procedure functie om de avg accuracy te berekenen
def model_finder():
  acc = -99999
  flag = 0
  for j in range(8, 20):
    model = Sequential()
    model.add(Dense(j, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_dim=8))
    model.add(Dense(6))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    calc_acc = kfold_procedure(X, y, model)
    print("calc_acc = " + str(calc_acc))
    print("acc = " + str(acc))
    # if j == 3:
    #   calc_acc = -10
    if calc_acc > acc:
      if flag == 1:
        flag = 0
      acc = calc_acc
    else:
      if flag == 1:
        print("Best accuracy: " + str(acc))
        print("Best L: " + str(j))
        return model
        exit(0)
      flag = 1
      print("FLAG SET AT ITERATION: " + str(j))
    K.clear_session()
    del model
  print("Best accuracy: " + str(acc))
  print("bigger is better")
  return model

model = tf.keras.models.load_model('model')

# uncomment dit als je nieuwe model wil fitten (en de oude overwriten)
# model = model_finder()
# model.evaluate(X_val, y_val, verbose=1)
# model.save('model')

# demonstrate prediction
yhat = model.predict(X_val, verbose=1)
yhat = np.array(yhat)

# Hier draai ik het normalizeren terug
for i in range(np.shape(yhat)[0]):
  yhat[i,:] = reading_input.reverse_normalize(yhat[i,:], min_arr[i], max_arr[i])

print(yhat)
# print("test loss, test acc:", results)
# plt.figure() 
# plt.plot(history.history['loss']) 
# plt.plot(history.history['val_loss']) 
# plt.title('model loss') 
# plt.ylabel('loss') 
# plt.xlabel('epoch') 
# plt.legend(['train', 'test'], loc='best') 
# plt.show() 
# plt.figure() 
# plt.plot(history.history['acc']) 
# plt.plot(history.history['val_acc']) 
# plt.title('model accuracy') 
# plt.ylabel('acc') 
# plt.xlabel('epoch') 
# plt.legend(['train', 'test'], loc='best') 
# plt.show()