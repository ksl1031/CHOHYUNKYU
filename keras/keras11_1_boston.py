from sklearn.datasets import load_boston # load_boston 을 가져온다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from sklearn.model_selection import train_test_split
import numpy as np

#1 데이터
datasets = load_boston() # 로드 보스턴을 데이터셋으로 부른다.
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size = 0.9, shuffle = True, random_state= 300)

# print(x)
# print(y)
# print(datasets)
print(datasets.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

print(datasets.DESCR)

print(x.shape,y.shape) # (506, 13) (506,)

# [실습]
#1. train_size 0.7
#2. R2 0.8이상

#2 모델 구성
model = Sequential()
model.add(Dense(60, input_dim = 13, activation = 'sigmoid'))
model.add(Dense(30,activation=LeakyReLU()))
model.add(Dense(40,activation=LeakyReLU()))
model.add(Dense(60,activation=LeakyReLU()))
model.add(Dense(20,activation=LeakyReLU()))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs = 600, batch_size = 70)

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 

print("r2스코어 : ",r2)