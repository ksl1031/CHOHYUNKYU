import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

#1 데이터
datasets = load_breast_cancer()

x = datasets['data']
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x, y,
                                                 train_size=0.7,
                                                 shuffle=True,
                                                 random_state=1234)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2 모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 30))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(monitor='val_loss',
              patience =20,
              mode = 'min',
              verbose=1,
              restore_best_weights=True,
              )
model.fit(x_train, y_train,
                 epochs = 2000,
                 batch_size = 200,
                 validation_split = 0.2,
                 verbose = 1,
                 callbacks=[es],
               
                 )

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)