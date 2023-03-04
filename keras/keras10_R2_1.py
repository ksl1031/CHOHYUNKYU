#1 R2를 음수가 아닌 0.5 이하로 만들것
#2 데이터는 건들지 말것
#3 레이어는 인풋 아웃풋 포함 7개 이상
#4 batch_size = 1
#5 히든레이어의 노드는 10개이상 100개 이하
#6 train_size = 75%
#7 epochs 100번이상
#8 loss지표는 mse, mae
#9 실습 시작

import numpy as np # 넘파이 위아래 상관없음
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train,x_test,y_train,y_test = train_test_split(x, y, # x = x_train, x_test # y = y_train, y_test로 분류 된다. x,y 위치가 바뀌어도 괜찮음
train_size=0.75,shuffle=True,random_state=10) # train_size 값을 변경해도됨 random_state 값을 변경해도됨

#2 모델 구성
model = Sequential()
model.add(Dense(20, input_dim = 1))
model.add(Dense(30))
model.add(Dense(35))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(32))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(11))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 500, batch_size = 1)

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict) 
print("r2스코어 : ",r2)