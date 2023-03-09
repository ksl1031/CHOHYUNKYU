import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


#1 데이터

# path = './_data/ddarung/'# '.' : 현재 폴더
path = './_data/_ddarung/'# path 변수에 데이터가 들어있는 경로를 알려준다.
path_save = './_save/ddarung/' # path 변수에 출력된 데이터를 저장하는 경로를 알려준다.

train_csv = pd.read_csv(path + 'train.csv', index_col = 0) # 인덱스 컬럼 0, 헤더와 인덱스는 연산 X, 행과 컬럼을 식별해준다.
# train_csv = pd.read_csv('./_data/ddarung/train.csv')

print(train_csv) # [1459 rows x 11 columns]
print(train_csv.shape) # (1459, 10)

test_csv = pd.read_csv(path + 'test.csv', index_col = 0) # 인덱스 컬럼 0, 헤더와 인덱스는 연산 X, 행과 컬럼을 식별해준다.

print(test_csv) # [715 rows x 9 columns]
print(test_csv.shape) # (715, 9)

#====================================================

# print(train_csv.columns)Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

print(train_csv.info()) # info : 정보 출력, 데이터의 행, 열, 데이터 타입, 널(null)값이 있는 열의 개수, 메모리 사용량 출력
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64

print(train_csv.describe()) # describe : 기초 통계 정보 출력 = count : 데이터 개수, mean : 평균값, std : 표준편차, min : 최소값, 25%, 50%, 75% : 사분위수, max : 최대값

print(type(train_csv)) # <class 'pandas.core.frame.DataFrame'>

#===================결측치 처리============================
# 결축치 제거 1. 제거
# print(train_csv.isnull())
print(train_csv.isnull().sum()) # isnull 결측치 여부확인 sum 결측치 값 갯수 확인 출력
train_csv = train_csv.dropna() # dropna : 결측치 제거
print(train_csv.isnull().sum()) # 제거된 결측치 값 재출력
print(train_csv.info())
print(train_csv.shape) # (1328, 10)


# ======================================= train_csv 데이터에서 x와y를 분리 ===========================
x = train_csv.drop(['count'], axis = 1) # count 컬럼을 제외한 나머지 데이터를 변수 x에 저장한다. drop : 지정한 컬럼을 제거한 결과를 반환한다. axis : 열방향으로 제거
print(x)
y = train_csv['count'] # 변수에서 제외한 count 컬럼을 y 변수에 저장한다.
print(y)
# ======================================= train_csv 데이터에서 x와y를 분리 ===========================

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.85,shuffle=True,random_state=2000)

print(x_train.shape, x_test.shape) # (1021, 9) (438, 9) -> (929, 9) (399, 9)
print(y_train.shape, y_test.shape) # (1021,) (438,) -> (929,) (399,)

#2 모델 구성
model = Sequential()
model.add(Dense(60, input_dim = 9))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(90, activation = 'relu'))
model.add(Dense(70, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 200, batch_size = 120, verbose = 1)

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict)
print("r2 스코어 : ", r2)

def RMSE(y_test,y_predict): # def : 함수를 정의할때 사용 ():안에 입력값을 받아서 
    return np.sqrt(mean_squared_error(y_test,y_predict)) # RMSE 함수 정의
rmse = RMSE(y_test, y_predict)                           # RMSE 함수 사용
print("RMSE : ", rmse)

#======================== submission.csv 를 만들어몹시다 ========================
# print(test_csv.isnull().sum()) # 여기도 경측치가 있다.
y_submit = model.predict(test_csv)
print(y_submit)

submission = pd.read_csv(path + 'submission.csv', index_col = 0)
print(submission)
submission['count'] = y_submit
print(submission)

submission.to_csv(path_save + 'submit_0307_0223.csv')


# 데이터
# path = './_data/ddarung/'
# train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
# test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
# train_csv = train_csv.dropna()
# x = train_csv.drop(['count'], axis = 1)
# y = train_csv['count']''