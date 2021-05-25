# MLP : Muli Layer Perceptron
#   Perceptron
import numpy as np
from tensorflow.python.keras import activations

# 1. data
x = np.array([[1,2,3,4,5,6,7,8,9,10],
            [11,12,13,14,15,16,17,18,19,20]]) # shape : (2, 10)
y = np.array([1,2,3,4,5,6,7,8,9,10])


    # <intend>
    # x   | 1   2   3   4   5   6   7   8   9   10
    #     | 11  12  13  14  15  16  17  18  19  20
    # ----|---------------------------------------
    # y   | 1   2   3   4   5   6   7   8   9   10

    # <transpose>
    # x      | y
    # -------|---
    # 1   11 | 1
    # 2   12 | 2
    # 3   13 | 3
    # 4   14 | 4
    # 5   15 | 5
    # 6   16 | 6
    # 7   17 | 7
    # 8   18 | 8
    # 9   19 | 9
    # 10  20 | 10


x = np.transpose(x) # shape : (10, 2), matching dimension

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
#model.add(Dense(10, input_dim=2))
model.add(Dense(10, input_shape=(2,))) # ignore row (ex) (1000, 10, 10, 3) -> input_shape=(10,10,3)
model.add(Dense(5))
model.add(Dense(1))

# 3. compile, trainnig
model.compile(loss='mse',optimizer='adam')
model.fit(x, y, batch_size=1, epochs=100, 
    validation_split=0.2)

# 4. evaluate, predict
# predict [[11,12,13],[21,22,23]]
y_predict = model.predict(np.transpose([[11,12,13],[21,22,23]]))
print('y_predict : ', y_predict)

# Epoch 100/100
# 8/8 [==============================] - 0s 2ms/step - loss: 6.8230e-05 - val_loss: 1.4660e-04
# y_predict :  [[10.983566]
#  [11.980624]
#  [12.977681]]