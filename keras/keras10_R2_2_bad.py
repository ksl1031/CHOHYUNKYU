from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np # 넘파이 위아래 상관없음

from sklearn.model_selection import train_test_split

#1 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train,x_test,y_train,y_test = train_test_split(x, y, # x = x_train, x_test # y = y_train, y_test로 분류 된다. x,y 위치가 바뀌어도 괜찮음
        train_size=0.8,shuffle=True,random_state=1234)