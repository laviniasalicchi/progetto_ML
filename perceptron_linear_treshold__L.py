import numpy as np


def net_func(x, weights):
    result = np.dot(x, weights)
    return result


def activation_function(net):
    if net > 0:
        return 1
    else:
        return 0


def out(net, activation):
    return activation(net)


def error(target, output):
    return target - output


def update_weights(weights, train, target):
    target = input_vector[1]
    if (output == input_vector[1]):
        return weights
    else:
        return weights + eta * target * train


def training(train, target, weights):
    for i in range(0, len(train[:, 0])):
        print(train[i])
        net = net_func(train[i], weights)
        if (net <= 0 and target[i] == 1):


train = np.array([[0, 0, 1],
                  [1, 0, 1],
                  [0, 1, 1],
                  [1, 1, 1]])
target_and = np.array([0, 0, 0, 1])
target_or = np.array([0, 1, 1, 1])

weights = np.random.rand(1, 3).T
eta = 0.02  # oracolo lavinia

training(train, target_and, weights)






