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

# # split a univariate sequence into samples
# def split_sequence(sequence, n_steps):
# 	X, y = list(), list()
# 	for i in range(len(sequence)):
# 		# find the end of this pattern
# 		end_ix = i + n_steps
# 		# check if we are beyond the sequence
# 		if end_ix > len(sequence)-1:
# 			break
# 		# gather input and output parts of the pattern
# 		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
# 		X.append(seq_x)
# 		y.append(seq_y)
# 	return array(X), array(y)


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
  X, y = list(), list()
  X.append(sequence[:41])
  y.append(sequence[-6:])
  return array(X), array(y)


reading_input = ReadingInput()
data = reading_input.process_data()
raw_seq = data[2,:] 
# choose a number of time steps
n_steps = 41
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# summarize the data
# for i in range(len(X)):
# 	print(X[i], y[i])
	
# define model
model = Sequential()
model.add(Dense(17, activation='relu', input_dim=41))
model.add(Dense(6))
model.compile(optimizer='adam', loss='mse')

history = model.fit(X, y, epochs=500, verbose=0, batch_size=1, shuffle=True, validation_split=0.8)

# demonstrate prediction
x_input = array([0.00000000e+00, 8.12702417e-04, 4.85195473e-05, 4.64574666e-03,
  3.78452469e-03, 1.85587268e-03, 1.94078189e-03, 6.30754115e-03,
  8.95185648e-03, 7.39923097e-03, 7.22941255e-03, 1.00920658e-02,
  1.37189020e-02, 1.10988464e-02, 1.86315062e-02, 1.44345653e-02,
  1.65451656e-02, 1.90560522e-02, 2.56304509e-02, 3.29205129e-02,
  4.88834439e-02, 5.86843925e-02, 5.44389321e-02, 4.93079900e-02,
  6.60108441e-02, 7.07900195e-02, 6.36697760e-02, 5.30318652e-02,
  6.98438884e-02, 7.95963174e-02, 7.94386288e-02, 1.09290280e-01,
  1.25859706e-01, 1.82093861e-01, 1.87236933e-01, 2.66129717e-01,
  3.59918002e-01, 3.63617617e-01, 4.43614221e-01, 5.80584903e-01,
  7.67785447e-01])
x_input = x_input.reshape((1, 41))
yhat = model.predict(x_input, verbose=0)

print("yhat")
print(yhat)

plt.figure() 
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('model loss') 
plt.ylabel('loss') 
plt.xlabel('epoch') 
plt.legend(['train', 'test'], loc='best') 
plt.show() 
plt.figure() 
plt.plot(history.history['acc']) 
plt.plot(history.history['val_acc']) 
plt.title('model accuracy') 
plt.ylabel('acc') 
plt.xlabel('epoch') 
plt.legend(['train', 'test'], loc='best') 
plt.show()

# [0.63146978 0.65216337 0.64123434 0.6981599  0.78877015 1.        ]

# train_start = 0
# n = 47
# train_end = int(np.floor(0.8*n))
# test_start = train_end
# test_end = n
# data_train = data[np.arange(train_start, train_end), :]
# data_test = data[np.arange(test_start, test_end), :]


# X_train = data_train[:, 1:]
# print(X_train)
# y_train = data_train[:, 0]
# print(y_train)
# X_test = data_test[:, 1:]
# y_test = data_test[:, 0]

# model = Sequential() 
# model.add(Dense(64, input_dim=30)) 
# model.add(BatchNormalization()) 
# model.add(LeakyReLU()) 
# model.add(Dense(2)) 
# model.add(Activation('softmax'))

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1) 
# opt = Adam(learning_rate=0.0001)
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])