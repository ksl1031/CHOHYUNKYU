import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1 데이터
path = './_data/ddarung/'
path_save = './_save/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

train_csv = train_csv.dropna()

x = train_csv(['count'], axis = 0)
y = train_csv['count']

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size= 0.7, shuffle=True,random_state=1234)

#2 모델 구성
model = Sequential()
model.add(Dense(6, input_dim = 9))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 300, batch_size = 5)

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)

def RMSE(y_test, y_predict):
    return pd.sqrt(mean_squared_error(y_test, y_predict))

y_submit = model.predict(test_csv)

submission = pd.read_csv(path + 'submission', index_col=0)
submission['count'] = y_submit
submission.to_csv(path_save + 'submission_0307_0520.csv')