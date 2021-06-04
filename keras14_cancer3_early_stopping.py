#이진 분류
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.callbacks import EarlyStopping

#1.
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(x[:5])
print(y[:5])
print(x.shape, y.shape)

print(dataset.feature_names)
print(dataset.DESCR) # 이진 분류 문제

## onehot encoding
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)
print(x_test[0])

#2.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(30,)))
model.add(Dense(2, activation='softmax'))

#3.
es = EarlyStopping(monitor='val_loss', patience=10, mode='min')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.1, callbacks=[es]) # early stopping

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

y_predict = model.predict(x_test)
# print("input: ",x_test[:5])
print("GT: ", y_test[:5])
print("predict: ", y_predict[:5])

# Epoch 45/100
# 409/409 [==============================] - 0s 427us/step - loss: 0.3270 - acc: 0.9242 - val_loss: 0.0571 - val_acc: 0.9783
# 4/4 [==============================] - 0s 634us/step - loss: 0.2973 - acc: 0.9123
# loss :  0.29732832312583923
# acc :  0.9122806787490845
# GT:  [[0. 1.]
#  [0. 1.]
#  [0. 1.]
#  [0. 1.]
#  [0. 1.]]
# predict:  [[6.0292851e-04 9.9939704e-01]
#  [2.8536475e-04 9.9971467e-01]
#  [1.6985410e-04 9.9983013e-01]
#  [7.0037966e-04 9.9929965e-01]
#  [3.2944898e-03 9.9670547e-01]]