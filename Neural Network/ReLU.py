import numpy as np
import matplotlib.pyplot as plt
# def ReLU(x):
#     result = np.zeros(x.shape)
#     result[x > 0] = x[x > 0]
#     return result

def ReLU(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = ReLU(x)
plt.title('ReLU')
plt.xlim(-5, 5)
plt.plot(x, y, linewidth=3)

plt.show()
