from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np # 넘파이 위아래 상관없음

from sklearn.model_selection import train_test_split

#1 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train,x_test,y_train,y_test = train_test_split(x, y, # x = x_train, x_test # y = y_train, y_test로 분류 된다. x,y 위치가 바뀌어도 괜찮음
        train_size=0.7,shuffle=True,random_state=1234)

#2 모델 구성
model = Sequential()
model.add(Dense(6, input_dim = 1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1, batch_size = 1)

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x)
# 시각화
import matplotlib.pyplot as plt

plt.scatter(x, y) # scatter : 점으로 그래프가 그려짐
plt.plot(x, y_predict, color = 'red') # plot : 점선으로 그래프가 그려짐
plt.show()