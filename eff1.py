###############   importing libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, Activation
import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras import optimizers
import matplotlib.pyplot as plt
###############   defineing fucntions
def pdata():
    dataset = pd.read_csv('.csv file path',names = ['Date','Open','High','Low','Close','Volume','OpenInt'])
    dataset = dataset.drop('Date',axis = 1)
    dataset = dataset.drop('OpenInt',axis = 1)
    dataset = dataset.drop(0,axis = 0)
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(dataset)
    global history_points
    history_points = 14
    ohlcv_histories_normalised = np.array([data_normalised[i  : i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.array([data_normalised[i + history_points][0].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)
    next_day_open_values = np.array([dataset['Open'][i + history_points] for i in range(len(dataset) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)
    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit( next_day_open_values ) 
    return ohlcv_histories_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser
def mse(real,predict):
    length = len(predict)
    msecalc = 0
    for i in range(length):
        msecalc += (real[i] - predict[i]) ** 2
    return msecalc / length
def n2l(ls):
    ls2 = []
    for i in ls:
        ls2.append(float(i[0]))
    return ls2
###############   preaparing normalised data to be traind
ohlcv_histories, next_day_open_values, unscaled_y, y_normaliser = pdata()
n = int(ohlcv_histories.shape[0] * 0.9)
ohlcv_train = ohlcv_histories[:n]
y_train = next_day_open_values[:n]
ohlcv_test = ohlcv_histories[n:]
y_test = next_day_open_values[n:]
unscaled_y_test = unscaled_y[n:]
###############   creating the model + traing + predicting + inverse nomarliesd to real number
model = Sequential()
model.add(LSTM(75))
model.add(Dense(64))
model.add(Activation('relu'))    # could be replaced by selu or tanh
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(optimizer='adam', loss='mse')
model.fit(x=ohlcv_train, y=y_train, batch_size=64, epochs=75)
y_test_predicted = model.predict(ohlcv_test)
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
###############   plotting the results
unscaled_y_test = n2l(unscaled_y_test.tolist())
y_test_predicted = n2l(y_test_predicted.tolist())
print(str("{:.2f}".format(mse(unscaled_y_test,y_test_predicted)*100))+'% Mistake in prediction')
plt.plot(unscaled_y_test, label='real',color = 'g')
plt.plot(y_test_predicted, label='predicted', color = 'r')
style = plt.gcf().set_size_inches(12,10)
plt.legend()
plt.show()
