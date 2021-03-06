# 학습단계 생략, 추론단계만 구현하기
from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784')
split_ratio = 0.9

n_train = int(mnist.data.shape[0] * split_ratio)
n_features = mnist.data.shape[1]
train_X = mnist.data[:n_train]
# test_X = mnist.data[n_train:]
train_y = mnist.target[:n_train]
# test_y = mnist.target[n_train:]

def Softmax(x):
    c = np.max(x)
    a = np.exp(x - c)
    sum_a = np.sum(a)
    return a / sum_a

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def init_network(k_list, n_features):
    """
    :param k_list: (list) hidden dim list
    :param n_features: input num features
    :return: (dict) weight and bias dictionary
    """
    network = {}
    network['w0'] = np.random.rand(n_features, k_list[0])
    network['b0'] = np.random.random(1)
    for i in range(1, len(k_list)):
        network[f'w{i}'] = np.random.rand(k_list[i-1], k_list[i])
        network[f'b{i}'] = np.random.random(1)

    return network

# batch 추가
def mlp(x, y, network, n_layers, batch_size = 32):
    accuracy_cnt = 0
    for j in range(0, len(x), batch_size):
        if j+batch_size >= len(x):
            x_batch = x[j:]
        else:
            x_batch = x[j:j + batch_size]

        for i in range(n_layers):
            x_batch = np.dot(x_batch, network[f'w{i}']) + network[f'b{i}']
            if i != n_layers - 1:
                x_batch = Sigmoid(x_batch)

        out = Softmax(x_batch)
        out = np.argmax(out, axis = 1)
        if j + batch_size >= len(x):
            accuracy_cnt += np.sum(out == y[j:])
        else:
            accuracy_cnt += np.sum(out == y[j:j + batch_size])

    return accuracy_cnt / len(x) * 100
k_list = [100, 50, 10]
n_layers = len(k_list)
network = init_network(k_list, n_features)
accuracy = mlp(train_X, train_y, network, n_layers)
print(f'{accuracy:.2f}%')





