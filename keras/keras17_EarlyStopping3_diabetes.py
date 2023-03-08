import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import load_diabetes

#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x, y,
                                                 train_size=0.7,
                                                 shuffle=True,
                                                 random_state=1234,
                                                 )

#2 모델 구성
model = Sequential()
model.add(Dense(6, input_dim = 10))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam',)
es = EarlyStopping(monitor='val_loss',
                   patience=20,
                   mode=min,
                   verbose=1,
                   restore_best_weights=True,
                   )
hist = model.fit(x_train,y_train,
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