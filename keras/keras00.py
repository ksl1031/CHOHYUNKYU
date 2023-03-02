#1 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])

#2 모델 구성
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(2, input_dim = 1))
<<<<<<< HEAD
model.add(Dense(4))
model.add(Dense(9))
=======
model.add(Dense(3))
model.add(Dense(9))
model.add(Dense(7))
>>>>>>> d3b79947e2b9013c1e999686008148f4f80fc30f
model.add(Dense(6))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = "mse", optimizer = "adam")
model.fit(x, y, epochs = 600)

<<<<<<< HEAD
#4 평가, 에측
=======
#4 평가, 예측
>>>>>>> d3b79947e2b9013c1e999686008148f4f80fc30f
loss = model.evaluate(x, y)
print("loss : ", loss)

result = model.predict([9])
<<<<<<< HEAD
print("9의 예측값 : ", result)
=======
print("9의 예측값 : ", result)

#1 8.979793
#2 8.995409
#3 8.999825
>>>>>>> d3b79947e2b9013c1e999686008148f4f80fc30f
