import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def step_function(x):
    y = x > 0
    return y.astype(int)

x = np.arange(-5, 5, 0.01)
y1 = sigmoid(x)
plt.plot(x, y1, color='red')
plt.ylim(-0.1, 1.1)
plt.title('Sigmoid VS Step')

y2 = step_function(x)

plt.plot(x, y2)
plt.show()