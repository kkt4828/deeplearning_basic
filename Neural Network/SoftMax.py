import numpy as np

def PrevSoftmax(x):
    exp_sum = np.sum(np.exp(x))
    return np.exp(x) / exp_sum

x = np.array([0.1, 0.6, 5])
print(PrevSoftmax(x))

#### 오버플로 발생

x = np.array([7000, 7020, 6990])
print(PrevSoftmax(x))

max_x = np.max(x)
print(PrevSoftmax(x - max_x))

### 최댓값을 빼주는 것을 적용한 softmax
def Softmax(x):
    max_x = np.max(x)
    new_x = np.exp(x - max_x)
    exp_sum = np.sum(new_x)
    return new_x / exp_sum
print(Softmax(x))