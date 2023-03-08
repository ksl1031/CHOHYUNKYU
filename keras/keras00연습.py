import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1 데이터
path = './_data/_ddarung/'
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.7,
                                                 shuffle=True,
                                                 random_state=1234,
                                                 )

#2 모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 9))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(monitor='val_loss',
                   patience=20,
                   mode='min',
                   verbose=1,
                   restore_best_weights=True,
                   )
hist = model.fit(x_train,y_train,
                 epochs=1000,
                 batch_size=5,
                 validation_split=0.3,
                 verbose=1,
                 callbacks=[es],
                 )
#4 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test,y_predict)
print("RMSE : ", rmse)

y_submit = model.predict(test_csv)

submission = pd.read_csv(path + 'submission.csv', index_col=0)
submission['count'] = y_submit
submission.to_csv(path_save + 'submit_0308_1054.csv')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize = (9,6))
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = '로스')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = '발_로스')
plt.title('보스턴')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()
plt.grid()
plt.show()