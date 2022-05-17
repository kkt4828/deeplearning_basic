import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.dot(x, w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.dot(x, w) + b
    if tmp > 0:
        return 1
    else:
        return 0

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.4

    tmp = np.dot(x, w) + b
    if tmp > 0:
        return 1
    else:
        return 0

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)

    y = AND(s1, s2)
    return y

print('XOR(0, 0)', XOR(0, 0))
print('XOR(0, 1)', XOR(0, 1))
print('XOR(1, 0)', XOR(1, 0))
print('XOR(1, 1)', XOR(1, 1))

