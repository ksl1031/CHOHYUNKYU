import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense ,Input, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

#1 데이터
path = './_data/_dacon_wine/'# '.' : 현재 폴더
path_save  = './_save/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
print(train_csv) # [5497 rows x 13 columns]
print(train_csv.shape) # (5497, 13)
print(test_csv) # [1000 rows x 12 columns]
print(test_csv.shape) # (1000, 12)

le = LabelEncoder()
le.fit(train_csv['type']) # type에 화이트,레드를 0,1로 변환시켜준다.
aaa = le.transform(train_csv['type'])
print(aaa)
print(type(aaa)) # <class 'numpy.ndarray'>
print(aaa.shape) # (5497,)
# print(np.unique(aaa, return_counts=True)) # (array([0, 1]), array([1338, 4159], dtype=int64))
train_csv['type'] = aaa
test_csv['type'] = le.transform(test_csv['type'])
print(le.transform(['red','white'])) # [0 1]
print(le.transform(['white','red'])) # [0 1]

x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']

y = pd.get_dummies(y)
y = np.array(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.85,
                                                 shuffle=True,
                                                 random_state=5555)

scaler = MinMaxScaler()
x_train_csv = scaler.fit_transform(x_train)
x_test_csv = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2 모델
input1 = Input(shape=(12,))
dense1 = Dense(200)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(500)(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(700)(drop2)
drop3 = Dropout(0.6)(dense3)
dense4 = Dense(300)(drop3)
output1 = Dense(7, activation = 'softmax')(dense4) # y의 라벨값만큼 아웃 레이어 출력
model = Model(inputs = input1, outputs = output1)

#3 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics='accuracy')
es = EarlyStopping(monitor='val_loss',
                   patience=30,
                   mode='min',
                   verbose=1,
                   restore_best_weights=True)
model.fit(x_train,y_train,
          epochs=10,
          batch_size=200,
          validation_split=0.2,
          verbose=1,
          callbacks=[es])

#4 평가, 예측
result = model.evaluate(x_test,y_test)
print("result : ", result)

y_predict = model.predict(x_test)

y_test_acc = np.argmax(y_test, axis = 1) # 각 행에 있는 열끼리 비교
y_predict = np.argmax(y_predict, axis = 1)

acc = accuracy_score(y_test_acc,y_predict)
print('acc : ', acc)
print(test_csv.shape)

y_submit = model.predict(test_csv)
print(y_submit.shape)
print(y_submit)

y_submit = np.argmax(y_submit, axis = 1)
y_submit += 3

submission = pd.read_csv(path + 'sample_submission.csv',index_col=0)
print(y_submit.shape)
print(y_submit)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

submission['quality'] = y_submit
submission.to_csv(path_save + 'sample_submission_' + date + '.csv')