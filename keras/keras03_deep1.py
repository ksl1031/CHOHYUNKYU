#1 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])

#2 모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(20))
model.add(Dense(3))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=100)

#loss1 : 1.1604
#loss2 : 0.0086
#loss3 : 0.0011