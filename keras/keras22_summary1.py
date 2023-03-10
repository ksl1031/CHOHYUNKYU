import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

#1 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2 모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(200000)) # boom
model.add(Dense(1))

model.summary()








