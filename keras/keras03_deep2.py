#1 데이터
import numpy as np # import 가져오다, as 별칭
x = np.array([1,2,3]) # array 배열
y = np.array([1,2,3])

#2 모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense # Dense 입출력 연결 레이어

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #loss 최소의, optimizer 최적의
model.fit(x,y,epochs=100) #훈련하다

#4 평가, 예측
loss = model.evaluate(x,y) #평가하다
print('loss : ', loss)

result = model.predict([4]) #예측하다
print("4의 예측값 : ", result)

# 4.010294
