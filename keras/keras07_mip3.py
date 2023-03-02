import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터
x = np.array(
    [[1,2,3,4,5,6,7,8,9,10],
     [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
     [9,8,7,6,5,4,3,2,1,0]]
)
y = np.array([11,12,13,14,15,16,17,18,19,20])
x = x.T # 행렬 반전

# 예측값 [[10, 1.4, 0]]

print(x.shape) # shape : 배열의 형태를 알아본다.
print(y.shape) # shape : 배열의 형태를 알아본다.

# 모델 구성
model = Sequential()
model.add(Dense(4, input_dim = 3)) # input_dim 입력 뉴런 숫자
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

# 컴파일, 훈련
model.compile(loss = "mse", optimizer = "adam") #loss 최소의, optimizer 최적의
model.fit(x, y, epochs = 600, batch_size=2) #epochs 전체 데이터 학습, batch_size : epochs를 나누어 실행하는 횟수

#평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)

result = model.predict([[10, 1.4, 0]])
print("10, 1.4, 0의 예측값 : ", result)
