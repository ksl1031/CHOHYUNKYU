from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1 데이터
x_train = np.array(range(1,17))
y_train = np.array(range(1,17))
# 실습 :: 짜르기
# train_test_split 로만 짜르기
# 10:3:3

x_train,x_test,y_train,y_test = train_test_split(x_train, y_train, train_size=13/16,random_state=1, shuffle=True)
print(x_train,y_train,x_test,y_test)
x_train,x_val,y_train,y_val = train_test_split(x_train, y_train,train_size=10/13,random_state=1, shuffle=True)
print(x_train,y_train,x_val,y_val,y_test,y_test)



'''
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
'''