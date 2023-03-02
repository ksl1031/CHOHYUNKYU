import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,9,8,7,6,5,4,3,2,1,]) # 마지막 숫자뒤에 ,가 있어도 문제 X

#실습 넘파이 리스트의 슬라이싱 7:3으로 잘라라
x_train = x[0:7] #[1,2,3,4,5,6,7]
x_train = x[:7] # : 앞에 숫자가 없으면 처음부터 시작
print(x_train) #[1,2,3,4,5,6,7]
x_test = x[7:10] #[8,9,10]
x_test = x[7:] # : 뒤에 숫자가 없으면 끝까지 계산
print(x_test) #[8,9,10]
y_train = y[0:7] #[10,9,8,7,6,5,4]
y_train = y[:7] # : 앞에 숫자가 없으면 처음부터 시작
print(y_train) #[10,9,8,7,6,5,4]
y_test = y[7:10] #[3,2,1]
y_test = y[7:] # : 뒤에 숫자가 없으면 끝까지 계산
print(y_test) #[3,2,1]
# 훈련은 범위내에서 훈련을 시켜야함

# print(x_train.shape, x_test.shape) #(7,)(3,)
# print(y_train.shape, y_test.shape) #(7,)(3,)