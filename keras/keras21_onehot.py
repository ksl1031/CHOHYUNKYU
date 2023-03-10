#[과제]
#3가지 원핫인코딩 방식을 비교할것

#1. pandas의 get_dummies
print(type(y))
print(y.shape)
y = pd.get_dummies(y)
print(type(y))
y = np.array(y)
print(y.shape)
print(type(y))
#2. keras의 to_categorical
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(y)
print(y.shape)
#3. sklearn의 OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = y.reshape(-1, 1)
y = ohe.fit_transform(y).toarray()
# 미세한 차이를 정리