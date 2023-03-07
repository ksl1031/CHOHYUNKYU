from sklearn.datasets import fetch_california_housing # 캘리포니아 주택 가격 데이터셋을 가져온다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from sklearn.model_selection import train_test_split
# [실습]
# 0.55 ~ 0.6

#1 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state=  20)

print(datasets.feature_names) # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)


#2 모델 구성
model = Sequential()
model.add(Dense(20, input_dim = 8, activation = 'sigmoid'))
model.add(Dense(30,activation=LeakyReLU()))
model.add(Dense(50,activation=LeakyReLU()))
model.add(Dense(60,activation=LeakyReLU()))
model.add(Dense(80,activation=LeakyReLU()))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs = 50, batch_size = 200)

#4 평가, 훈련
loss = model.evaluate(x_test, y_test)
print("loss", loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

# r2스코어 : 0.597406860400812