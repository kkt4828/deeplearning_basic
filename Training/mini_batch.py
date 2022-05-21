import numpy as np
import pandas as pd

mnist_train = pd.read_csv('../data/mnist/train.csv/train.csv')
x_train = np.array(mnist_train.iloc[:,1:])
y_train = np.array(mnist_train.iloc[:,0])
train_size = x_train.shape[0]
batch_size = 32
batch_masks = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_masks]
y_batch = y_train[batch_masks]

def CrossEntropyLoss(y_pred, y_target, eps = 0.001):
    if y_pred.ndim == 1:
        return -(np.sum(y_target.reshape(1, y_target.size) * np.log(y_pred.reshape(1, y_pred.size) + eps)))

    return -np.mean((np.sum(y_target * np.log(y_pred + eps), axis = 1)))
target = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

pred = np.array([[0.1, 0.4, 0, 0.8, 0.2, .3, .1, .05, .9, .2], [0.1, 0.4, 0, 0.8, 0.2, .3, .1, .05, .9, .2]])

print(CrossEntropyLoss(pred, target))
