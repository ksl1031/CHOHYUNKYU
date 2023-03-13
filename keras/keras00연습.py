import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

#1 데이터
path = './_data/_dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.7,
                                                 shuffle=True,
                                                 random_state=1212)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2 모델
input1 = Input(shape=(8))
dense1 = Dense(6)(input1)
dense2 = Dense(6)(dense1)
dense3 = Dense(6)(dense2)
dense4 = Dense(6)(dense3)
output1 = Dense(1)(dense4)
model = Model(outputs = output1, inputs = input1)

#3 컴파일, 훈련
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics='accuracy')
es = EarlyStopping(monitor='val_loss',
                   patience=20,
                   mode='min',
                   verbose=1,
                   restore_best_weights=True)
model.fit(x_train,y_train,
          epochs=1000,
          batch_size = 20,
          validation_split=0.2,
          verbose = 1,
          callbacks=[es])

#4 평가, 예측
result = model.evaluate(x_test,y_test)
print("result : ", result)

y_predict = np.round(model.predict(x_test))

acc = accuracy_score(y_test,y_predict)
print("acc : ", acc)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)

y_submit = np.round(model.predict(test_csv))

submission = pd.read_csv(path + 'sample_submission.csv', index_col =0)
submission['Outcome'] = y_submit
submission.to_csv(path_save + 'sample_submission_0313_1757.csv')







