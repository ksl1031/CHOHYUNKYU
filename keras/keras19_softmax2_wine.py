<<<<<<< HEAD
from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split


# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

#  ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 
#   'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
# print(datasets.DESCR)
# print(x.shape, y.shape)       # (178, 13), (178,)

"""print(np.unique(y))
from keras.utils import to_categorical
y = to_categorical(y)
print(y.shape)
"""

y = pd.get_dummies(y)
print(y.shape)

"""
from sklearn.preprocessing import OneHotEncoder
oh = OneHotEncoder()
y = oh.fit_transform(y)
print(y.shape)"""

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    random_state=123,
                                                    )


# 2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=13))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'],
              )
model.fit(x_train, y_train,
          epochs=100,
          batch_size=10,
          verbose=1,
          validation_split=0.2,
          )

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = np.round(model.predict(x_test))

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)
=======
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score


#1 데이터
datasets = load_wine()
print(datasets.DESCR)         # 판다스 describe()
print(datasets.feature_names) # 판다스 clolumns()

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


#판다스에 갯더미, 사이킷런에 원핫인코더
#=========================================================
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 shuffle=True,
                                                #  random_state=333,
                                                 train_size=0.5, # 데이터의 비율을 일정하게 조절한다.
                                                 stratify=y
                                                 )
print(y_train) # [1 0 2 0 1 1 0 0 2 0 1 1 1 2 0]
print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2]), array([5, 5, 5], dtype=int64))

#2 모델 구성
model = Sequential()
model.add(Dense(50, activation='relu',input_dim=13))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(3, activation='softmax')) # y의 라벨 갯수만큼 아웃레이어 출력 

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
# accuracy_score를 사용해서 스코어를 빼세요.

'''#4 평가, 예측
result = model.evaluate(x_test,y_test) # loss 값, metrics 값 출력
print("result : ", result)

y_predict = model.predict(x_test)
print(y_predict.shape)

y_predict = np.argmax(y_predict, axis= 1)
print(y_predict.shape)

y_true = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true,y_predict)
print('acc : ', acc)'''

# =================================
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
>>>>>>> b1c62c7924a9aed68538356d32e2ca2524a7aa58
