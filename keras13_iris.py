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
print(dataset.DESCR) # 회귀 문제

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
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, batch_size=5, epochs=100)

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('results : ', results)

y_predict = model.predict(x_test)
print("input: ",x_test[:5])
print("GT: ", y_test[:5])
print("predict: ", y_predict[:5])

# 1/1 [==============================] - 0s 114ms/step - loss: 0.0732 - mae: 0.9333
# results :  [0.07319959253072739, 0.9333333969116211]
# input:  [[6.2 2.9 4.3 1.3]
#  [6.8 2.8 4.8 1.4]
#  [6.5 2.8 4.6 1.5]
#  [5.7 3.8 1.7 0.3]
#  [6.6 2.9 4.6 1.3]]
# GT:  [1 1 1 0 1]
# predict:  [[1.03307313e-04 9.98898745e-01 9.97900381e-04]
#  [1.10666406e-05 9.90435779e-01 9.55307763e-03]
#  [1.26821260e-05 9.78787005e-01 2.12002788e-02]
#  [9.99879122e-01 1.20830395e-04 7.85878693e-21]
#  [4.18381569e-05 9.98566210e-01 1.39206671e-03]]