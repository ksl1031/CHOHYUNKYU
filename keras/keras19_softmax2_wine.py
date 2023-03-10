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