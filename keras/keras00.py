#1데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])

#2모델 구성
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(2, input_dim = 1))
model.add(Dense(4))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

#3컴파일 ,훈련
model.compile(loss = "mse", optimizer = "adam")
model.fit(x, y, epochs = 600)

#4평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

reslut = model.predict([7])
print("7의 예측값은 : ", reslut)

#1 7.000001
#2 6.9999423
#3 6.9558496