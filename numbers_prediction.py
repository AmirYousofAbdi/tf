from tensorflow import keras
from keras.layers import Dense ,Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt
dataset = keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = dataset.load_data()
x_tarin = keras.utils.normalize(x_train,axis=1)
x_test = keras.utils.normalize(x_test,axis=1)
model = Sequential()
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='sigmoid'))
model.add(Dense(128,activation='selu'))
model.add(Dense(128,activation='tanh'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
model.fit(x_train,y_train,epochs = 5)
pred = model.predict([x_test])
y_pred = []
for ind1 in range(len(pred)):
    MAX = max(pred[ind])
    for ind2 in range(len(pred[ind1])):
        if MAX == pred[ind1][ind2]:
            y_pred.append(ind2)
INCORRECT_pred = 100 - (len([i for i in range(len(y_pred)) if y_pred[i]!=y_test[i]]) / 100)
print(INCORRECT_pred)
