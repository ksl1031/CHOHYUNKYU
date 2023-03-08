import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping


#1 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (506, 13) (506,)

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.7, shuffle=True,random_state=20)

#2 모델 구성
model = Sequential()
model.add(Dense(50, activation = 'sigmoid', input_dim = 13))
model.add(Dense(50, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(20, activation = 'sigmoid'))
model.add(Dense(80, activation = 'sigmoid'))
model.add(Dense(15))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(monitor='val_loss', # 지정한 데이터의 값을 모니터링 한다.
                   patience=20, # 기준값 손실값이 지정한 숫자만큼 연속으로 감소하지안으면 학습 중단시킨다.
                   mode = 'min', # mode의 defualt 값 auto 값이 최소값이면 min, 값이 최대값이면 max
                   verbose=1,
                   restore_best_weights=True, # defualt 값 False
                   )
hist = model.fit(x_train,y_train, 
                 epochs = 2000,
                 batch_size = 20,
                 validation_split = 0.2,
                 verbose = 1,
                 callbacks=[es],# fit에 훈련 저장, callbacks : 지정한 값을 호출
                 )

# print("============================")
# print(hist)
# print("============================")
# print(hist.history) # epochs 돌린 만큼 값이 저장된다.
# print("============================")
# print(hist.history['loss']) # epochs 돌린 만큼 loss를 저장 값을 출력한다.
# print("============================")
print(hist.history['val_loss']) # epochs 돌린 만큰 val_loss를 저장 값을 출력한다.

#4 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("r2 스코어 : ", r2)

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