import numpy as np
from sklearn.datasets import load_breast_cancer # 유방암 데이터
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score,accuracy_score # 정확도 점수

#1 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR) # 판다스 : .describe()
print(datasets.feature_names) # 판다스 : .columns()

x = datasets['data']
y = datasets.target

print(x.shape,y.shape) # (569, 30) (569,)
# print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 shuffle=True,
                                                 random_state=200,
                                                 test_size=0.2,
                                                 )

#2 모델 구성
model =Sequential()
model.add(Dense(70, activation = 'linear',input_dim = 30))
model.add(Dense(20, activation = 'linear'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(50, activation = 'linear'))
model.add(Dense(1, activation = 'sigmoid')) # sigmoid : 0 ~ 1

#3 컴파일, 훈련
model.compile(loss= 'binary_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy','mse'], # 모델의 성능을 모니터링 해준다. acc,mse,mean_squared_error 보여줄수있다. list : 2개 이상을 보여줄수있다.
              )
es = EarlyStopping(monitor='val_accuracy', # 이진분류 loss : 'binary_crossentropy'
                   patience=20,
                   mode='min',
                   verbose=1,
                   restore_best_weights=True,
                   )
hist : model.fit(x_train,y_train,
          epochs=1000,
          batch_size = 20,
          validation_split=0.2,
          verbose =1,
          callbacks=[es],
          )

#4 평가, 예측
result = model.evaluate(x_test,y_test) # loss 값, metrics 값 출력
print("result : ", result)

y_predict = np.round(model.predict(x_test)) # np.round:반올림 / 예측값을 반올림해서 0,1의 값이 나올 수 있도록해줌 (0.5까지는 0으로, 0.6부터는 1로)
print("===================================")
print(y_test[:5])
print(y_predict[:5])
print(np.round(y_predict[:5]))

# print("===================================")

acc = accuracy_score(y_test,y_predict)
print('acc : ', acc)