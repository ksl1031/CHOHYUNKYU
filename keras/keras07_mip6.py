import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21, 31), range(201,211)]) # range : 특정구간의 숫자 범위를 만들어준다.
print(x) # 컴퓨터 숫자는 0부터 시작 : 10 -1
x = x.T # (10, 3)
print(x.shape) # (10, 3)  shape : 배열의 형태를 알아본다.

y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
              [9,8,7,6,5,4,3,2,1,0]]) # (3, 10)
y = y.T # (10, 3) # T : transpose
print(y.shape) # (10, 3) shape : 몇행 몇열인지 반환

# 예측 [[9, 30, 210]] -> 예상 y값 [[10, 1.9, 0]]

#2 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3)) # input_dim 입력 뉴런 숫자
model.add(Dense(6))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))

#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #loss 최소의, optimizer 최적의
model.fit(x,y,epochs=10, batch_size=4) # epochs 전체 데이터 학습, batch_size : epochs를 나누어 실행하는 횟수

#4평가, 예측
loss = model.evaluate(x,y) # 평가하다.
print("loss : ",loss)

result = model.predict([[9, 30 , 210]]) # 예측하다.
print("[9, 30 ,210]의 예측값 : ]", result)