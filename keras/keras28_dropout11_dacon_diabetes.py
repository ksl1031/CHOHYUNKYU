import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler

#1 데이터
path = './_data/_dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv)
x = train_csv.drop(['Outcome'], axis = 1)
y = train_csv['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.75,
                                                 shuffle=True,
                                                 random_state=200)

# scaler = MinMaxScaler() # 0 ~ 1
# scaler = StandardScaler() # 0, 1
# scaler = MaxAbsScaler() # -1, 1
# scaler = RobustScaler() # 25% ~ 75%
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)


#2 모델 구성
# model = Sequential() # 노트를 설정한 값만큼 빼고 evaluate 에서 모두 계산한다.
# model.add(Dense(30, input_shape=(8,))) # 8스칼라 1벡터
# model.add(Dropout(0.3))
# model.add(Dense(20))
# model.add(Dropout(0.2))
# model.add(Dense(10))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.summary()

input1 = Input(shape = (8,))
dense1 = Dense(300, activation = 'relu')(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(200, activation = 'relu')(drop1)
drop2 = Dropout(0.5)(dense2)
dense3 = Dense(100, activation = 'relu')(drop2)
drop3 = Dropout(0.5)(dense3)
output1 = Dense(1, activation = 'sigmoid')(drop3)
model = Model(inputs = input1, outputs = output1)

#3 컴파일,훈련
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam',
              metrics='accuracy')
es = EarlyStopping(monitor='val_loss',
                   patience=20,
                   mode = 'min',
                   verbose = 1,
                   restore_best_weights=True,
                   )
model.fit(x_train, y_train,
          epochs = 1000,
          batch_size = 20,
          validation_split = 0.5,
          verbose = 1,
          callbacks=[es],
          )

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = np.round(model.predict(x_test))

acc = accuracy_score(y_test,y_predict)
print("acc : ", acc)

y_submit = np.round(model.predict(test_csv))

submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Outcome'] = y_submit
submission.to_csv(path_save + 'sample_submission_0314_1317.csv')