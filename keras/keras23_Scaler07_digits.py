from sklearn.datasets import load_digits
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (1797, 64), (1797,)

y = to_categorical(y)
print(y.shape)     # (1797, 64), (1797,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    stratify=y,
                                                    )

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(y_train)
print(np.unique(y_train, return_counts=True))

# 2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'],
              )
model.fit(x_train, y_train,
          epochs=100,
          batch_size=10,
          validation_split=0.2,
          )

# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = np.round(model.predict(x_test))
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)