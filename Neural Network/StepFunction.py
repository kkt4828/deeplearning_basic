## 입력으로 실수값만 받음
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

print(step_function(3), step_function(-1))

## Numpy 배열도 받을 수 있게 하자
import numpy as np
def step_function(x):
    y = x > 0
    return y.astype(int)

print(step_function(np.array([3 ,- 1, 5])))

## 그래프를 그려보자
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.01)
y = step_function(x)

plt.plot(x, y)
plt.show()