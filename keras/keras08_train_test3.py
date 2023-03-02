import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense




#1 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10]) # 마지막 숫자뒤에 ,가 있어도 문제 X


# [검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법
# 힌트 사이킷런

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(
    x, y,
    train_size = 0.7,
    test_size = 0.3, # 1이 넘으면 안된다.
    random_state=1234,
    shuffle = True,
)
print(x_train)
print(y_test)

#2 모델 구성
model = Sequential()
model.add(Dense(1, input_dim = 1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 2)

#4 예측, 평가
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
result = model.predict([11])
print("11의 예측값 : ", result)
