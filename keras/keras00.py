from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([4,5,6,3,2,1,7,8,10,9])

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random = 1)

#2 모델 구성
model = Sequential()
model.add(Dense(6, input_dim = 1))
model.add(Dense(8))
model.add(Dense(9))
model.add(Dense(6))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 300, batch_size = 1)

#4 평가, 예측
loss = model.predict(x_test, y_test)
print("loss = ", loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(x, y)
print("r2스코어 : ", r2)