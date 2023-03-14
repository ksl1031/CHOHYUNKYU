import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

#1 데이터
path = './_data/_kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

train_csv = train_csv.dropna()

x = train_csv.drop(['count','casual','registered'], axis = 1)
y = train_csv['count']

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.8,
                                                 shuffle=True,
                                                 random_state=131)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2 모델 구성
model = Sequential()
model.add(Dense(60, input_dim = 8))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(90))
model.add(Dense(70,activation = 'relu'))
model.add(Dense(1))

#3 컴파일,훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
es = EarlyStopping(monitor='val_loss',
                   patience=30,
                   mode = 'min',
                   verbose = 1,
                   restore_best_weights=True,
                   )
model.fit(x_train, y_train,
          epochs = 1000,
          batch_size = 50,
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

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)

y_submit = model.predict(test_csv)

submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
submission['count'] = y_submit
submission.to_csv(path_save + 'submit_0312_1307.csv')