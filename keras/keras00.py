import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,19,13,14,6,7,8,9,3,4,12])

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 20)

#2 모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(25))
model.add(Dense(13))
model.add(Dense(17))
model.add(Dense(21))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 300, batch_size = 1)

#4 평가, 훈련
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)