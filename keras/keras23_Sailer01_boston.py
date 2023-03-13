from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 전처리
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

print(type(x)) # <class 'numpy.ndarray'>
print(x)

# print(np.min(x), np.max(x)) # 0.0 711.0
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# print(np.min(x), np.max(x)) # 0.0 1.0

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.8,
                                                 random_state=333,
                                                 )

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# 하나를 선택해서 사용할수 있다.
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # train 에 변한 비율에 맞춰진다.

print(np.min(x_train), np.max(x_train)) # 0.0 1.0000000000000002
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test)) # -0.00557837618540494 1.1478180091225065


'''#2 모델 구성
model = Sequential()
model.add(Dense(1, input_dim = 13))

#3 컴파일, 훈련
model.compile(loss = 'mse',
              optimizer = 'adam',
              )
model.fit(x_train,y_train,
          epochs = 10,
          )

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)'''
















