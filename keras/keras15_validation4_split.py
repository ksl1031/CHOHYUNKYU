from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1 데이터
x = np.array(range(1,17))
y= np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)



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
model.fit(x_train,y_train, epochs=100, batch_size=10, validation_split = 0.2) # split : 훈련 데이터 비율을 설정한 만큼 평가한다.

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)

result = model.predict([17])
print("17의 예측값 : ", result)
