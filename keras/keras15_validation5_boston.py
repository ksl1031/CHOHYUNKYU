from sklearn.datasets import load_boston 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state= 300)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=300)

#2 모델 구성
model = Sequential()
model.add(Dense(60, input_dim = 13, activation = 'sigmoid'))
model.add(Dense(30,activation=LeakyReLU()))
model.add(Dense(40,activation=LeakyReLU()))
model.add(Dense(60,activation=LeakyReLU()))
model.add(Dense(20,activation=LeakyReLU()))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs = 500, batch_size = 0,validation_data = (x_val, y_val)) # _split

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict) 

print("r2스코어 : ",r2)