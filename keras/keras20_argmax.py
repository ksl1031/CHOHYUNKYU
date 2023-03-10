import numpy as np

a = np.array([[1,2,3],[6,4,5],[7,9,2],[3,2,1],[2,3,1]])
print(a)
print(a.shape) #(5,3)
print(np.argmax(a)) #전체 개수중에 가장 높은 숫자를 출력 7
print(np.argmax(a, axis=0)) # [2 2 1] 0은 행이다. 행끼리 비교
print(np.argmax(a, axis=1)) # [2 0 1 0 1] 1은 열이다. 열끼리 비교
print(np.argmax(a, axis=-1)) # [2 0 1 0 1] -1은 가장 마지막
                             # 가장 마지막 축
                             # 2차원 이니까 가장 마지막 축은 1
                             # 그래서 -1 쓰면 데이터는 1과 동일