import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score

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
                                                 random_state=1122)

#2 모델
model = Sequential()
model.add(Dense(6, input_dim = 8))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#3 컴파일, 훈련
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics='accuracy')
es = EarlyStopping(monitor='val_loss',
                   patience=30,
                   mode='min',
                   verbose=1,
                   restore_best_weights=True)
model.fit(x_train,y_train,
          epochs=1000,
          batch_size=1,
          validation_split=0.2,
          verbose=1,
          callbacks=[es])

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)

y_predict = np.round(model.predict(x_test))

r2 = r2_score(y_test,y_predict)
print("r2 스코어 : ", r2)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)

acc = accuracy_score(y_test,y_predict)
print("acc : ", acc)

y_submit = np.round(model.predict(test_csv))

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Outcome'] = y_submit
submission.to_csv(path_save + 'sample_submission_0312_2012.csv')


