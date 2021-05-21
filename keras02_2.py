import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. data
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([2,4,6,8,10,12,14,16,18,20])

x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([111,112,113,114,115,116,117,118,119,120])

x_predict = np.array([111,112,113])

# 2. model
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(9))
model.add(Dense(9))
model.add(Dense(9))
model.add(Dense(9))
model.add(Dense(3))


# 3. compile, training
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=1)

# 4. evaluate, predict
loss = model.evaluate(x_test, y_test, batch_size=1)
print('loss : ', loss)

results = model.predict(x_predict)
print('results : ', results)