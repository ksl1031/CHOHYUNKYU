import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 데이터
x = np.array([[1,2],
             [3,4],
             [5,6],
             [7,8],
             [9,10]])
print(x)
print(x.shape)

#2 모델 구성
#3 컴파일, 훈련
#4 평가, 예측