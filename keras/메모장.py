import numpy as np
import pandas as pd
import matplotlib.pyplot as plt : 그래프를 그릴 수 있는 라이브러리
import time
from tensorflow.python.keras.models import Sequential : 시퀀셜 모델 불러오기
from tensorflow.python.keras.layers import Dense, LeakyReLU : 댄스 레이어 불러오기,
from tensorflow.python.keras.utils import to_categorical : 
from tensorflow.python.keras.callbacks import EarlyStopping : 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder

warning : 경고, 작동은 한다.
error : 작동 안함
행무시 열우선, 열=컬럼=특성=피쳐
import : 불러오다
from : 가져오다
as : 별칭
array : 배열
transpose : 행렬 반전
Sequential :
Dense : 입출력 연결 레이어
add : 모델에 대한 호출
input_dim : 입력 데이터의 개수를 지정하는 매개변수
input layer : 신경망의 입력층
hidden layer : 신경망의 입력층과 출력층 사이에 위치한 층
output layer : 신경망의 출력층
compile : 기계어로 바꾼다.
fit : 모델을 훈련을 시키는 함수 최적화된 가중치(weight)를 찾는다.
loss : 최소의
optiimzer : 최적의
mse : 평균 제곱 오차 = 모델의 예측값과 실제 값 사이의 차이를 평가하는 손실 함수
mae : 절대값
adam : 최적화 알고리즘
epochs : 데이터를 몇번 반복해서 훈련할지 결정한다.
batch : 일괄
batch_size : 데이터를 쪼개서 작업함 default값은 32
shape : 데이터의 행렬을 반환
evaluate : 모델의 성능을 평가
result : 결과
predict : 모델의 입력 데이터를 예측
train : 모델 학습 과정
test : 모델의 최종적으로 평가
scatter : 데이터를 점으로 시각화 해주는 함수
plot : 데이터를 선으로 시각화 해주는 함수
show : 화면에 그림을 출력 해주는 함수
r2_score : 결정 계수 0과 1 사이의 값으로 나타냄 1에 가까울수록 실제데이터와 일치
LeakyReLU : 활성화 함수
datasets.feature_names : 데이터셋의 특성
datasets.DESCR : 데이터셋의 정보
verbose : 0 = 아무것도 안나온다. 1,auto = 다 보인다. 2 = 프로그래스바만 없어짐 3,4,5 = epochs만 나온다.  default값은 1
'.' : 폴더 지정
info : 정보 출력, 데이터의 행, 열, 데이터 타입, 널(null)값이 있는 열의 개수, 메모리 사용량 출력
describe : 기초 통계 정보 출력 = count : 데이터 개수, mean : 평균값, std : 표준편차, min : 최소값, 25%, 50%, 75% : 사분위수, max : 최대값
isnull :  결측치 여부확인
sum : 결측치 값 갯수 확인 출력
dropna : 결측치 제거
drop : 지정한 컬럼을 제거한 결과를 반환한다.
axis : 0,1 행열방향으로 제거
def : 함수를 정의할때 사용 ():안에 입력값을 받아서
x = train,test,val,predict
y = train,test,val, X
클래스 - 함수 차이 메일로 보내기
리스트 : 2개 이상, 딕셔너리 : 키 벨류, 튜플 메일로 보내기
회귀
loss : mse,mae
out layer : linear, y의 컬럼의 갯수만큼
one hot : 없음
분류
loss : 이중 = binary_crossentropy, 다중 = categorical_crossentropy
out layer : 이중 = sigmoid, 1, 다중 = softmax, y의 값(라벨,클래스)의 갯수만큼
one hot : 이중 = 없음, 다중 = 있음
데이콘  캐글 디아뱃 MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler 주말에 메일로 보내기
정규화 : normalization, 장점 = 과부하 X 성능,속도 좋아질수도 있다. 동일한 비율로 변경이 된다. 최대값으로 나눈다. 훈련(train) 데이터만 정규화한다.
0 ~ 100, 80개 0 ~ 1
80 ~ 110, 20개 0.8 ~ 1.1
훈련(train) 데이터의 범위의 비율에 맞춰서 나머지 데이터도 똑같이 한다.
train과 test 분리후 정규화를 한다.

from keras.utils import to_categorical
y = to_categorical(y)
print(y.shape)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y)
print(y.shape)

print(y.shape)
y = pd.get_dummies(y)
print(type(y))
y = np.array(y)
print(y.shape)

scaler = MinMaxScaler()
scaler = StandardScaler()
scaler = MaxAbsScaler()
scaler = RobustScaler()
============================================================
1.1 경로, 가져오기
1.2 확인사항 5가지 : shape, columns, info(), describe(), type
1.3 결측지 제거
print(train_csv.isnull().sum()) 결측치 확인
train_csv = trina_csv.dropna() 결측치 제거
print(train_csv.isnull().sum()) 결측치 재확인
1.4 라벨인코더
le = LabelEncoder()
le.fit(train_csv['type'])
aaa = le.transform(train_csv['type'])
train_csv['type'] = aaa
test_csv['type'] = le.transform(test_csv['type'])
print(train_csv)
print(test_csv)
print(train_csv.shape)
print(test_csv.shape)
1.5 x, y 분리
x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']
1.6 원핫인코딩
y = pd.get_dummies(y)
y = np.array(y)
1.8 scaler
==============================================================
print(train_csv.shape) 데이터프레임의 행과 열의 개수를 출력합니다
print(train_csv.columns) 데이터프레임의 열 이름을 출력합니다.
print(train_csv.info()) 데이터프레임의 정보를 출력합니다. 
      각 열의 데이터 타입과 누락된 데이터가 있는지 여부 등을 확인할 수 있습니다.
print(train_csv.describe()) 데이터프레임의 요약 통계 정보를 출력합니다.
            수치형 열의 평균, 표준편차, 최소값, 최대값 등을 확인할 수 있습니다.
print(type(train_csv)) 변수의 데이터 타입을 출력합니다.

