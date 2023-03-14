import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

#1 데이터
path = './_data/_dacon_wine/'
path_save = './_save/dacon_wine.'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['quality','type'],axis=1)
y = train_csv['quality']

y = pd.get_dummies(y)
y = np.array(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.7,
                                                 shuffle=True,
                                                 random_state=1212)

scaler = MinMaxScaler()
scaler = MaxAbsScaler()
scaler = StandardScaler()
scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv.drop('type',axis=1))

#2 모델
input1 = Input(shape=(11))
dense1 = Dense(20)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(20)(drop1)
dense3 = Dense(20)(dense2)
drop2 = Dropout(0.3)(dense3)
dense4 = Dense(20)(drop2)
output1 = Dense(7)(dense4)
model = Model(inputs = input1, outputs = output1)

#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics='accuracy')
es = EarlyStopping(monitor='val_loss',
                   patience=20,
                   mode='min',
                   verbose=1,
                   restore_best_weights=True)
model.fit(x_train,y_train,
          epochs=1000,
          batch_size=2000,
          validation_split=0.2,
          verbose=1,
          callbacks=[es])

#4 평가, 예측
result = model.evaluate(x_test,y_test)
print("result : ", result)

y_predict = model.predict(x_test)

y_test_acc = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)

acc = accuracy_score(y_test_acc,y_predict)
print("acc : ", acc)

y_submit = model.predict(test_csv)
y_submit = np.argmax(y_submit, axis=1)
y_submit += 3

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['quality'] = y_submit
submission.to_csv(path_save + 'submission_0314_1919.csv')

