import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,5,4])

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #loss 최소의, optimizer 최적의
model.fit(x,y,epochs=100) #훈련하다

#4 평가, 예측
loss = model.evaluate(x,y) #평가하다
print("loss : ", loss)

result = model.predict([6]) #예측하다
print("6의 예측값 : ", result) # 결과