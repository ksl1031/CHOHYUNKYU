from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model # 함수형 모델, 다른점 모델 정의를 상단 or 하단
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler # 전처리
from sklearn.metrics import r2_score

#1 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

print(type(x)) # <class 'numpy.ndarray'>
print(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.8,
                                                 random_state=333,
                                                 )

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# 하나를 선택해서 사용할수 있다.
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train) # fit을 같이 써도 된다. scaler.fit
x_test = scaler.transform(x_test) # train 에 변한 비율에 맞춰진다.

#2 모델 구성
# model = Sequential()
# model.add(Dense(30, input_shape=(13,), name = 'S1')) # 13스칼라 1벡터
# model.add(Dense(20, name = 'S2'))
# model.add(Dense(10, name = 'S3'))
# model.add(Dense(1, name = 'S4'))
# model.summary()

input1 = Input(shape = (13,), name = 'h1')
danse1 = Dense(30, name = 'h2')(input1)
danse2 = Dense(20, name = 'h3')(danse1)
danse3 = Dense(10, name = 'h4')(danse2)
output1 = Dense(1, name = 'h5')(danse3)
model = Model(inputs = input1, outputs = output1)

# model.save('./_save/keras26_1_save_model.h5') # 모델 구조만 저장이 된다.


# 데이터가 3차원이면(시계열 데이터)
# (1000, 100, 1) ->>> input_shape=(100, 1) 행 무시
# 데이터가 4차원이면(이미지 데이터)
# (60000, 32, 32, 3) ->>> input_shape=(32, 32, 3) 행 무시

#3 컴파일, 훈련
model.compile(loss = 'mse',
              optimizer = 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss',
                   patience=10,
                   mode = 'min',
                   verbose=1,
                   restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',
                      mode='auto', # mode 디폴트 값 min
                      verbose=1,
                      save_best_only=True, # 가장 좋은 값을 저장
                      filepath='./_save/MCP/keras27_ModelCheckPoint1.hdf5') # filepath : 파일의 경로를 나타내는 문자열 = 현재 경로에 파일 생성 저장

model.fit(x_train,y_train, 
                 epochs = 1000,
                 batch_size = 20,
                 validation_split = 0.2,
                 verbose = 1,
                 callbacks=[es, mcp])

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)