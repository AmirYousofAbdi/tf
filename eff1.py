import keras
from keras.models import Sequential
from keras.layers import Dense,Activation
import pandas as pd
import numpy as np
#from keras import optimizers
from sklearn import preprocessing

def pdata():
    url = 'C:\\Users\\Lenovo\\Desktop\\dtt2.csv'
    names = ['Open','High','Low','Close','Volume']
    dataset = pd.read_csv(url,names = names)
    dataset = dataset.drop(0,axis = 0)
    data_normaliser = preprocessing.MinMaxScaler()
    datan = data_normaliser.fit_transform(dataset)
    history_points = 20
    ohlcvhn = ohlcv_histories_normalised = np.array([datan[i  : i + history_points].copy() for i in range(len(datan) - history_points)])
    ndovn = np.array([datan[:, 0][i + history_points].copy() for i in range(len(datan) - history_points)])
    #ndovn = np.expand_dims(ndovn, -1)
    
    
    return ohlcvhn,ndovn

ohlcvhn,ndovn = pdata()
test_split = 0.9
n = int(ohlcvhn.shape[0] * test_split)
ohlcvhn,ndovn = ohlcvhn[:n],ndovn[:n]
print(ohlcvhn)
model = Sequential()
model.add(Dense(850, input_dim=n, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(150, activation='softmax'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.add(Activation('linear'))   #khati
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(optimizer='adam', loss='mse')
model.fit(x=ohlcvhn, y=ndovn,  epochs=100, batch_size=64)