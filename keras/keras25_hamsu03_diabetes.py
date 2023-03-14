import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler

#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x, y,
                                                 train_size=0.7,
                                                 shuffle=True,
                                                 random_state=1234,
                                                 )

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2 모델 구성
# model = Sequential()
# model.add(Dense(6, input_dim = 10))
# model.add(Dense(7))
# model.add(Dense(7))
# model.add(Dense(7))
# model.add(Dense(7))
# model.add(Dense(1))

input1 = Input(shape = (10,))
danse1 = Dense(6)(input1)
danse2 = Dense(7)(danse1)
danse3 = Dense(7)(danse2)
danse4 = Dense(7)(danse3)
danse5 = Dense(7)(danse4)
output1 = Dense(1)(danse5)
model = Model(inputs = input1, outputs = output1)

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam',)
es = EarlyStopping(monitor='val_loss',
                   patience=20,
                   mode=min,
                   verbose=1,
                   restore_best_weights=True,
                   )
model.fit(x_train,y_train,
          epochs = 200,
          batch_size=5,
          validation_split = 0.2,
          verbose = 1,
          callbacks=[es],
          )

#4 평가, 예측
loss = model.evaluate(x_test,y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)
