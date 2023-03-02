import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터
x = np.array(
    [[1, 1],
    [2, 1],
    [3, 1],
    [4, 1],
    [5, 2],
    [6, 1.3],
    [7, 1.4],
    [8, 1.5],
    [9, 1.6],
    [10, 1.4]]
) # (10,2)
y = np.array([11,12,13,14,15,16,17,18,19,20])

print(x.shape) #(10, 2) -> 2개의 특성을 가진 10개의 데이터  shape : 배열의 형태를 알아본다.
print(y.shape) #(10, )  shape : 배열의 형태를 알아본다.

#2 모델구성
model = Sequential()
model.add(Dense(4, input_dim=2)) # input_dim 입력 뉴런 숫자
model.add(Dense(6))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #loss 최소의, optimizer 최적의
model.fit(x,y,epochs=30, batch_size=3)

#4평가, 예측
loss = model.evaluate(x,y)
print("loss : ",loss)

result = model.predict([[10, 1.4]])
print("[10, 1.4]의 예측값 : ]", result)