import numpy as np


# activation function sigmoid
# deriv per poter utilizzare in futuro la f'(net)
def sigmoid(x, deriv=False):

    a = 1
    if (deriv is True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def net_func(x, weights):

    result = np.dot(x, weights)
    return result[0][0]


def out(net, activation):

    return (activation(net))


def error(target, output):

    return (target-output)


def delta(error, net):

    return -2 * error * sigmoid(net, True)


x = np.array([[1, 2, 3, 4, 5, 6]])
# y = np.array([[1,2,3,4,5,6]])

# print('x+y',x+y)
weights = np.random.rand(1, 6).T     # T = trasposta
print('weights', weights)
y = 8
eta = 0.02
err = [5.0]
i = 0
net = net_func(x, weights)

print(net)
print(type(sigmoid(net, True)))
print(type(err[0]))
for i in range(2):
    i = i+1
    print('iterazione ', i)
    net = net_func(x, weights)
    # print('net', net)
    output = out(net, sigmoid)
    err = error(y, output)**2
    # print('err',err)
    delt = delta(error(y, output), net)
    print('delta', delt)
    # print('old', weights)
    weights = weights + eta*delt*x
    print('error', err)
    # print(eta*delt)
    # print(eta*delt*x)
# print('new',weights)
print('weights', weights)
print()