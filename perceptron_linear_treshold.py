# ==============================================================================
# E' un Perceptron, che cazzo ti aspettavi.
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================

import numpy as np

def net_func(x, weights):
    result = np.dot(x, weights)
    return result

#Linear threshold unit
def activation_function(net):
    if net > 0:
        return 1
    else:
        return -1

def out(net, activation):
    return activation(net)

def error(target, output):
    return (target - output)

def training(train, target, weights, eta):

    epoch_weights = weights
    for i in range(0, len(train[:,0])):
        #print(train[i])
        net = net_func(train[i], epoch_weights)
        print('net:\t',net)
        output = activation_function(net)
        if (output == target[i]):
            #do nothing
            print('pesi immutati', output - target[i])
            #return weights
        else:
            epoch_weights_new = np.add(epoch_weights, np.multiply(eta * (target[i] - output), train[i]))
            epoch_weights = epoch_weights_new
            print('epoch_weights_new', epoch_weights_new)
    return epoch_weights

'''
-1 corrisponde all'uscita logica 0
è necessario usare -1 e 1 per evitare che i pesi siano sempre positivi
'''
train = np.array([[0, 0, 1],
                  [1, 0, 1],
                  [0, 1, 1],
                  [1, 1, 1]])
target_and = np.array([-1, -1, -1, 1])
target_or = np.array([-1, 1, 1, 1])

#starting_weights = (np.random.rand(1,3))[0]
starting_weights = np.zeros((1,3), dtype=np.int)[0]
eta = 0.02     #oracolo lavinia

i = 0
while True: #in teoria sarebbe finchè i pesi non cambiano
    weights_new = training(train, target_or, starting_weights, eta)
    if np.array_equal(starting_weights, weights_new):
        print('Ho capito! Ho capito! Basta!')
        break
    starting_weights = weights_new
    print('epoch:', i)
    i += 1

#test del modello
for i in range(0, len(train[:,0])):
    print('Expected:', 1 if target_or[i] > 0 else 0, end='')
    net = net_func(train[i], starting_weights)
    out = activation_function(net)
    print('\tPredicted:', 1 if out > 0 else 0,end='')
    print()
