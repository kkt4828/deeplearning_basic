import numpy as np

def ReLU(x):
    return np.maximum(0, x)
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def init_weight():
    w1 = np.array([[1, 3, 5], [2, 4, 6]])
    w2 = np.array([[3, 4], [5, 6], [7, 8]])
    w3 = np.array([[-5, 3], [-2, 6]])
    b1 = -5
    b2 = -4
    b3 = 3

    return {'w1' : w1, 'w2': w2, 'w3' : w3, 'b1': b1, 'b2' : b2, 'b3' : b3}

def Layer3_NeuralNetwork(w_dict, x):
    a1 = np.dot(x, w_dict['w1']) + w_dict['b1']
    z1 = Sigmoid(a1)

    a2 = np.dot(z1, w_dict['w2']) + w_dict['b2']
    z2 = Sigmoid(a2)

    a3 = np.dot(z2, w_dict['w3']) + w_dict['b3']
    z3 = ReLU(a3)

    return a3, z3

X = np.array([1, 2])
w_dict = init_weight()

print(Layer3_NeuralNetwork(w_dict, X))









