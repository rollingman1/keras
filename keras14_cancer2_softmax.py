#이진 분류
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_breast_cancer

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
model.add(Dense(1, activation='softmax'))

#3.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x, y, epochs=50, batch_size=1, validation_split=0.1)

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

y_predict = model.predict(x_test)
# print("input: ",x_test[:5])
print("GT: ", y_test[:5])
print("predict: ", y_predict[:5])
