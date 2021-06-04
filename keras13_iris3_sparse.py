import numpy as np

# 1. 데이터
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target
print(x[:5])
print(y[:5])
print(x.shape, y.shape) # (150, 4) (150,)

print(dataset.feature_names)
print(dataset.DESCR) # 다중 분류 문제

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=66
)
print(x_test[0])

# 2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(4,))
h1 = Dense(10)(input1)
h2 = Dense(10)(h1)
h3 = Dense(10)(h2)
h4 = Dense(5)(h3)
output1 =  Dense(3, activation='softmax')(h4)
model = Model(inputs=input1, outputs=output1)

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=5, epochs=100)

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

y_predict = model.predict(x_test)
print("input: ",x_test[:5])
print("GT: ", y_test[:5])
print("predict: ", y_predict[:5])

# loss :  0.10458726435899734
# acc :  0.9666666388511658
# input:  [[6.2 2.9 4.3 1.3]
#  [6.8 2.8 4.8 1.4]
#  [6.5 2.8 4.6 1.5]
#  [5.7 3.8 1.7 0.3]
#  [6.6 2.9 4.6 1.3]]
# GT:  [1 1 1 0 1]
# predict:  [[1.8589883e-04 9.9738389e-01 2.4301766e-03]
#  [2.0468111e-05 9.6618205e-01 3.3797398e-02]
#  [2.8679267e-05 9.5601761e-01 4.3953631e-02]
#  [9.9984729e-01 1.5268635e-04 3.3992467e-20]
#  [7.1334711e-05 9.9491239e-01 5.0163087e-03]]