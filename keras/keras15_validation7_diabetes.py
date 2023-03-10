from sklearn.datasets import load_diabetes # 당뇨병 환자의 데이터셋을 가져온다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size = 0.65, shuffle = True, random_state= 600)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=300)

#2 모델 구성
model = Sequential()
model.add(Dense(100, input_dim = 10, activation = 'sigmoid'))
model.add(Dense(70,activation=LeakyReLU()))
model.add(Dense(80,activation=LeakyReLU()))
model.add(Dense(60,activation=LeakyReLU()))
model.add(Dense(70,activation=LeakyReLU()))
model.add(Dense(90,activation=LeakyReLU()))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 900, batch_size = 0,validation_data = (x_val, y_val))

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)