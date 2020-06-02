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
import numpy
import random
import ReadingInput

def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	return df


# create a differenced series
def difference(dataset, interval=1):
  diff = list()
  print(dataset)
  for i in range(interval, len(dataset)):
    value = dataset[i] - dataset[i - interval]
    diff.append(value)


def fit_model(train, batch_size, nb_epoch, neurons):
  X = train[:len(train)//2]
  y = train[len(train)//2:]
  # X, y = train[:, 0:-1], train[:, -1]
  model = Sequential()
  model.add(Dense(neurons, activation='relu', input_dim=X.shape[1]))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=0, shuffle=False)
  return model


# run a repeated experiment
def experiment(repeats, series, epochs, lag, neurons):
  # transform data to be stationary
  raw_values = series
  diff_values = difference(raw_values, 1)
  # transform data to be supervised learning
  supervised = timeseries_to_supervised(diff_values, lag)
  supervised_values = supervised.values[lag:,:]
  # split data into train and test-sets
  train, test = supervised_values[0:-12], supervised_values[-12:]
  # transform the scale of the data
  ## scaler, train_scaled, test_scaled = scale(train, test)
  # run experiment
  error_scores = list()
  for r in range(repeats):
    # fit the model
    batch_size = 4
    ## train_trimmed = train_scaled[2:, :]
    model = fit_model(train, batch_size, epochs, neurons)
    # forecast test dataset
    ## test_reshaped = test_scaled[:,0:-1]
    output = model.predict(test, batch_size=batch_size)
    predictions = list()
    for i in range(len(output)):
      yhat = output[i,0]
      ## X = test_scaled[i, 0:-1]
      # invert scaling
      ## yhat = invert_scale(scaler, X, yhat)
      # invert differencing
      ## yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
      # store forecast
      predictions.append(yhat)
    # report performance
    rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
    print('%d) Test RMSE: %.3f' % (r+1, rmse))
    error_scores.append(rmse)
  return error_scores
 


def Main():
  data = [random.uniform(0,1) for x in range(0,47)]
  data = df(data)
  # data = data.T
  repeats = 30
  results = DataFrame()
  lag = 1
  neurons = 1
  epochs = [50, 100, 500, 1000, 2000]
  for e in epochs:
    results[str(e)] = experiment(repeats, data, e, lag, neurons)
  print(results.describe)
  results.boxplot()
  pyplot.savefig('boxplot_epochs.png')
  # print(timeseries_to_supervised(data))



if __name__ == "__main__":
  Main()