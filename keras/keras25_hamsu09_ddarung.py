import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

#1 데이터
path = './_data/_ddarung/'
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

train_csv = train_csv.dropna()
x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,
                                                 shuffle=True,
                                                 random_state=20,
                                                 )

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2 모델 구성
# model = Sequential()
# model.add(Dense(500, input_dim = 9))
# model.add(Dense(700,activation= 'linear'))
# model.add(Dense(66,activation= 'linear'))
# model.add(Dense(605,activation= 'relu'))
# model.add(Dense(7000,activation= 'relu'))
# model.add(Dense(1))

input1 = Input(shape = (9,))
danse1 = Dense(500)(input1)
danse2 = Dense(700, activation='linear')(danse1)
danse3 = Dense(66, activation='linear')(danse2)
danse4 = Dense(605, activation='relu')(danse3)
danse5 = Dense(7000, activation='relu')(danse4)
output1 = Dense(1, activation='softmax')(danse5)
model = Model(inputs = input1, outputs = output1)

#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam')
es = EarlyStopping(monitor = 'val_loss',
                   patience = 20,
                   mode = 'min',
                   verbose = 1,
                   restore_best_weights=True,
                   )
model.fit(x_train,y_train,
          epochs = 1000,
          batch_size = 50,
          validation_split=0.2,
          verbose=1,
          callbacks =[es])

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

y_submit = model.predict(test_csv)

submission = pd.read_csv(path + 'submission.csv', index_col = 0)
submission['count'] = y_submit
submission.to_csv(path_save + 'submit_0314_1439.csv')