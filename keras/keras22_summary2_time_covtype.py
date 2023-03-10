import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score

#1 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1, 1)
y = ohe.fit_transform(y).toarray()

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape) # (581012, 8)

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 shuffle=True,
                                                #  random_state=333,
                                                 train_size=0.5, # 데이터의 비율을 일정하게 조절한다.
                                                 stratify=y
                                                 )
print(y_train) # [1 0 2 0 1 1 0 0 2 0 1 1 1 2 0]
print(np.unique(y_train, return_counts=True)) #(array([0., 1.], dtype=float32), array([2033542,  290506], dtype=int64))

#2 모델 구성
model = Sequential()
model.add(Dense(50, activation='relu',input_dim=54))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(7, activation='softmax')) # y의 라벨 갯수만큼 아웃레이어 출력

# model.summary() # Total params: 7,767, 모델의 레이어, 출력 형태, 파라미터 수 출력

#3 컴파일, 훈련
model.compile(loss ='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['acc'],
              )
import time
start_time = time.time() # 현시점의 시간을 반환한다.
model.fit(x_train,y_train,
          epochs = 10,
          batch_size = 300,
          validation_split=0.2,
          verbose=1,
          )
end_time = time.time() # 현시점의 시간을 반환한다.

#4 평가, 예측
result = model.evaluate(x_test,y_test)
print("result : ", result)
print("loss : ", result[0]) # result의 0번째 값
print("acc : ", result[1]) # result의 1번째 값

print("걸린 시간은 : ",(end_time - start_time)) # 숫자를 반올림 해준다.

'''y_pred = model.predict(x_test)

y_test_acc = np.argmax(y_test, axis = 1) # 각 행에 있는 열끼리 비교
y_pred = np.argmax(y_pred, axis = 1)

acc = accuracy_score(y_test_acc,y_pred)
print('accuracy_score : ', acc)'''