import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1 데이터
path = './_data/_dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
print(train_csv)
print(train_csv.isnull().sum()) # 결측치 확인
train_csv = train_csv.dropna()

x = train_csv.drop(['Outcome'], axis = 1)
y = train_csv['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 test_size=0.2,
                                                 shuffle=True,
                                                 random_state=3421,
                                                 )


#2 모델 구성
model = Sequential()
model.add(Dense(500,activation = 'relu',input_dim = 8))
model.add(Dense(600,activation = 'relu'))
model.add(Dense(7000,activation = 'relu'))
model.add(Dense(80,activation = 'relu'))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#3 컴파일, 훈련
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics= 'accuracy',
              )
es = EarlyStopping(monitor='val_loss',
                   patience=30,
                   mode='min',
                   verbose=1,
                   restore_best_weights=True,
                   )
model.fit(x_train,y_train,
          epochs = 1000,
          batch_size = 400,
          validation_split=0.2,
          verbose = 1,
          callbacks=[es],
          )

#4 평가, 예측
result = model.evaluate(x_test, y_test)
print("result : ", result)

y_predict = np.round(model.predict(x_test))

# print(y_test[:5])
# print(y_predict[:5])
# print(np.round(y_predict[:5]))

acc = accuracy_score(y_test,y_predict)
print("acc : ", acc)

y_submit = np.round(model.predict(test_csv))

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Outcome'] = y_submit
submission.to_csv(path_save + 'sample_submission_0310_2245.csv')