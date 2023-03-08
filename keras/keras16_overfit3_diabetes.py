from sklearn.datasets import load_diabetes # 당뇨병 환자의 데이터셋을 가져온다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size = 0.65, shuffle = True, random_state= 600)

print(datasets.feature_names) # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)


print(x.shape, y.shape) # (442, 10) (442,)


#2 모델 구성
model = Sequential()
model.add(Dense(100, input_dim = 10, activation = 'sigmoid'))
model.add(Dense(10,activation= 'relu'))
model.add(Dense(30,activation= 'relu'))
model.add(Dense(60,activation= 'relu'))
model.add(Dense(20,activation= 'relu'))
model.add(Dense(60,activation= 'relu'))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
hist = model.fit(x_train, y_train, epochs = 900, batch_size = 0)

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize = (9,6))
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = '로스')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = '발_로스')
plt.title('보스턴')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()
plt.grid()
plt.show()