import numpy as np

def MSE(y_pred, y_target):
    return 0.5 * np.sum((y_pred - y_target) ** 2)

target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

pred = np.array([0.1, 0.4, 0, 0.8, 0.2, .3, .1, .05, .9, .2])

print(MSE(pred, target))

def CrossEntropy(y_pred, y_target, eps = 0.001):
    return -(np.sum(y_target * np.log(y_pred + eps)))

print(CrossEntropy(pred, target))
import matplotlib.pyplot as plt
x = np.arange(0, 1, 0.01)
y = np.log(x)

plt.plot(x, y)
plt.title("Log Function")
plt.show()