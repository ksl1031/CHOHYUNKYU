# 저장할때 평가결과값, 훈련시간등을 파일에 저장 
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model # 함수형 모델, 다른점 모델 정의를 상단 or 하단
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler # 전처리
from sklearn.metrics import r2_score

#1 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']

print(type(x)) # <class 'numpy.ndarray'>
print(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.7,
                                                 random_state=20,
                                                 )

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# 하나를 선택해서 사용할수 있다.
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train) # fit을 같이 써도 된다. scaler.fit
x_test = scaler.transform(x_test) # train 에 변한 비율에 맞춰진다.

#2 모델 구성
model = Sequential() # 노트를 설정한 값만큼 빼고 evaluate 에서 모두 계산한다.
model.add(Dense(30, input_shape=(54,), name = 'S1')) # 54스칼라 1벡터
model.add(Dropout(0.3))
model.add(Dense(20, name = 'S2'))
model.add(Dropout(0.2))
model.add(Dense(10, name = 'S3'))
model.add(Dropout(0.5))
model.add(Dense(7, name = 'S4'))
model.summary()

# input1 = Input(shape = (54,), name = 'h1')
# danse1 = Dense(30, name = 'h2')(input1)
# drop1 = Dropout(0.3)(danse1)
# danse2 = Dense(20, name = 'h3')(drop1)
# drop2 = Dropout(0.2)(danse2)
# danse3 = Dense(10, name = 'h4')(drop2)
# drop3 = Dropout(0.5)(danse3)
# output1 = Dense(7, name = 'h5')(drop3)
# model = Model(inputs = input1, outputs = output1)

# model.save('./_save/keras26_1_save_model.h5') # 모델 구조만 저장이 된다.


# 데이터가 3차원이면(시계열 데이터)
# (1000, 100, 1) ->>> input_shape=(100, 1) 행 무시
# 데이터가 4차원이면(이미지 데이터)
# (60000, 32, 32, 3) ->>> input_shape=(32, 32, 3) 행 무시

#3 컴파일, 훈련
model.compile(loss = 'mse',
              optimizer = 'adam')

import datetime
date = datetime.datetime.now() # 현재시간을 date에 넣어준다.
print(date)
# 2023-03-14 11:11:37.181155
date = date.strftime("%m%d_%H%M") # 시간을 문자로 바꾼다. 월 일 시 분 반환해준다.
print(date) # 0314_1115

filepath = './_save/MCP/keras27_4/' # filpath를 현재 파일 경로로 지정
filename = '{epoch:04d}-{val_loss:4f}.hdf5'

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',
                   patience=10,
                   mode = 'min',
                   verbose=1,)
                #    restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto', # mode 디폴트 값 min
                      verbose=1,
                      save_best_only=True, # 가장 좋은 값을 저장
                      filepath="".join([filepath, "k27_", date, "_", filename])) # join : 빈공간에 무언가를 합친다.

model.fit(x_train,y_train, 
                 epochs = 1000,
                 batch_size = 20,
                 validation_split = 0.2,
                 verbose = 1,
                 callbacks=[es, ]) # mcp


# model.save('./_save/MCP/keras27_3_save_model.h5') # 모델과 가중치 저장

#4 평가, 예측
print("====================== 1. 기본 출력 ===========================")
loss = model.evaluate(x_test, y_test,verbose=0) # loss 출력 값 제거
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict)
print("r2 스코어 : ", r2)

