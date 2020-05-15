from easygui import multenterbox as mulent
from easygui import msgbox 
import pandas as pd
import datetime
import pandas_datareader.data as web
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, Activation
import numpy as np
from sklearn import preprocessing
from keras import optimizers
import matplotlib.pyplot as plt
def CORRECT_DATA(DATA_INF):
    try:
        DATA_NAME = DATA_INF[0]
        if DATA_NAME.upper() == 'CLOSE':
            return 'See you soon'
        DATA_INF[1] = list(map(lambda x:int(x),(DATA_INF[1].split('/'))))
        DATA_INF[2] = list(map(lambda x:int(x),(DATA_INF[2].split('/'))))
        START = datetime.datetime(DATA_INF[1][0],DATA_INF[1][1],DATA_INF[1][2])
        END = datetime.datetime(DATA_INF[2][0],DATA_INF[2][1],DATA_INF[2][2])
        df = web.DataReader(DATA_NAME,'yahoo',START,END)
        FILE_NAME = DATA_NAME + 'dt.csv'
        df.to_csv(FILE_NAME)
        return FILE_NAME
    except:
        msgbox('Enter again your stock information correctly!')
        CORRECT_DATA(mulent('Enter your data information.(Date must be like:YY/MM/DD)','Stock prediction',['Stock name','Start','End']))
def pdata():
    df = pd.read_csv(CORRECT_DATA(mulent('Enter your data information.(Date must be like:YY/MM/DD)','Stock prediction',['Stock name','Start','End'])),names = ['Date','Open','High','Low','Close','Volume','Adj Close'])
    df = df.drop('Date',axis = 1)
    df = df.drop('Adj Close',axis = 1)
    df = df.drop(0,axis = 0)
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(df)
    history_points = 14
    ohlcv_histories_normalised = np.array([data_normalised[i  : i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.array([data_normalised[i + history_points][0].copy() for i in range(len(data_normalised) - history_points)])
    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)
    next_day_open_values = np.array([df['Open'][i + history_points] for i in range(len(dataset) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)
    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit( next_day_open_values ) 
    return history_points,ohlcv_histories_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser
def n2l(ls):
    ls2 = []
    for i in ls:
        ls2.append(float(i[0]))
    return ls2
###############   preaparing normalised data to be traind
history_points,ohlcv_histories, next_day_open_values, unscaled_y, y_normaliser = pdata()
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
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(optimizer='adam', loss='mse')
model.fit(x=ohlcv_train, y=y_train, batch_size=64, epochs=75)
y_test_predicted = model.predict(ohlcv_test)
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
###############   plotting the results
unscaled_y_test = n2l(unscaled_y_test.tolist())
y_test_predicted = n2l(y_test_predicted.tolist())
plt.plot(unscaled_y_test, label='real',color = 'g')
plt.plot(y_test_predicted, label='predicted', color = 'r')
style = plt.gcf().set_size_inches(12,10)
plt.legend()
plt.show()
