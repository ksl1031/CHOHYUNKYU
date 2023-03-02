import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21, 31), range(201,211)]) # range : 특정구간의 숫자 범위를 만들어준다.
print(x) # 컴퓨터 숫자는 0부터 시작 : 10 -1
x = x.T # (10, 3)
print(x.shape) # (10, 3)  shape : 배열의 형태를 알아본다.


y = np.array(([1,2,3,4,5,6,7,8,9,10])) # (1, 10)
y = y.T # (10, 1)
print(y.shape) # shape : 배열의 형태를 알아본다.

#2 모델구성
model = Sequential()
model.add(Dense(4, input_dim=3)) # input_dim 입력 뉴런 숫자
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #loss 최소의, optimizer 최적의
model.fit(x,y,epochs=30, batch_size=2) #epochs 전체 데이터 학습, batch_size : epochs를 나누어 실행하는 횟수

#4평가, 예측
loss = model.evaluate(x,y)
print("loss : ",loss)

result = model.predict([[9, 30 ,210]])
print("[9, 30 ,210]의 예측값 : ]", result)