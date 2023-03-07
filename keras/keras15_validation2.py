from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1 데이터
x_train = np.array(range(1,17)) # [ 1  2  3  4  5  6  7  8  9 10], (10, )
y_train = np.array(range(1,17))

# x_val = np.array([14,15,16]) # validation : train 데이터중에 일부를 평가를 한다.
# y_val = np.array([14,15,16])

# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])
# 실습 :: 짜르기

x_val = x_train[13:17]
y_val = x_train[13:17]
print(x_val)
print(y_val)

x_test = x_train[10:13]
y_test = x_train[10:13]
print(x_test)
print(y_test)


#2 모델
model = Sequential()
model.add(Dense(5, activation="linear", input_dim =1))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(90))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(1))

# #3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=10, validation_data = (x_val, y_val))

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)

result = model.predict([17])
print("17의 예측값 : ", result)