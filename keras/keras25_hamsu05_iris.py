import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

#1 데이터
datasets = load_iris()
print(datasets.DESCR)         # 판다스 describe()
print(datasets.feature_names) # 판다스 clolumns()
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets['target']
print(x.shape,y.shape) # (150, 4) (150,)
print(x)
print(y)
print('y의 라벨값 : ', np.unique(y)) # y의 라벨값 : [0 1 2]

#=======================여기에서 원핫을 한다.===============
print(y)
print(y.shape)
print(type(y))
y = pd.get_dummies(y)
print(type(y))
y = np.array(y)
print(y.shape)

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# # print(y)
# print(y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 shuffle=True,
                                                #  random_state=333,
                                                 train_size=0.5,
                                                 stratify=y, # 데이터의 비율을 일정하게 조절한다.
                                                 )

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2 모델 구성
# model = Sequential()
# model.add(Dense(50, activation='relu',input_dim=4))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(3, activation='softmax')) # y의 라벨 갯수만큼 아웃레이어 출력 

input1 = Input(shape = (4,))
danse1 = Dense(50, activation='relu')(input1)
danse2 = Dense(40, activation='relu')(danse1)
danse3 = Dense(30, activation='relu')(danse2)
danse4 = Dense(20, activation='relu')(danse3)
danse5 = Dense(40, activation='relu')(danse4)
output1 = Dense(3, activation='softmax')(danse5)
model = Model(inputs = input1, outputs = output1)

#3 컴파일, 훈련
model.compile(loss ='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['acc'],
              )
model.fit(x_train,y_train,
          epochs = 200,
          batch_size = 10,
          validation_split=0.525,
          verbose=1,
          )

#4 평가, 예측
result = model.evaluate(x_test,y_test)
print("result : ", result)
print("loss : ", result[0]) # result의 0번째 값
print("acc : ", result[1]) # result의 1번째 값

y_pred = model.predict(x_test)
# print(y_test.shape)
# print(y_pred.shape)
# print(y_test[:5])
# print(y_pred[:5])

y_test_acc = np.argmax(y_test, axis = 1) # 각 행에 있는 열끼리 비교
y_pred = np.argmax(y_pred, axis = 1)

acc = accuracy_score(y_test_acc,y_pred)
print('accuracy_score : ', acc)