# -*- coding: utf-8 -*-

import numpy as np

from layer import Layer
from input_layer import InputLayer
from hidden_layer import HiddenLayer
from output_layer import OutputLayer
from neural_net import NeuralNetwork



def __main__():
    ''' ---- parte per importare il dataset esterno ---- '''
    filename = 'ML-CUP17-TR.csv'
    raw_data = open(filename, 'r')

    data = np.loadtxt(raw_data, delimiter=",")

    x = np.empty([data.shape[0], data.shape[1] - 3])
    target_x = np.empty([data.shape[0], 1])
    target_y = np.empty([data.shape[0], 1])
    target_values = np.empty([data.shape[0],    2])   # // target values = (pattern, target_x/y)



    for i in range(0, len(data[:, 0])):
        k = 0
        for j in range(1,11):
            x[i][k] = data[i][j]
            k = k+1
        target_x[i][0] = data[i][11]
        target_y[i][0] = data[i][12]
        target_values[i][0] = data[i][11]
        target_values[i][1] = data[i][12]


    ''' ---- parte per importare il dataset esterno ---- '''


    target_values = np.transpose(target_values)
    print('target_values.shape', target_values.shape)

    x = np.transpose(x)
    input_layer = InputLayer(x.shape[0])
    input_layer.create_weights(x.shape[0])
    print('x shape', x.shape)
    print('input_layer', input_layer.weights.shape)
    print('input_layer n_units', input_layer.n_units)


    #hidden_layer = HiddenLayer(input_layer.n_units[1])
    hidden_layer = HiddenLayer(5)
    hidden_layer.create_weights(input_layer.n_units)
    hidden_layer.set_activation_function('sigmoid')

    print('h_layer n_units', hidden_layer.n_units)

    '''hidden_layer2 = HiddenLayer(4)
    hidden_layer2.create_weights(hidden_layer.n_units)
    hidden_layer2.set_activation_function('sigmoid')'''

    output_layer = OutputLayer(2)
    output_layer.create_weights(hidden_layer.n_units)
    output_layer.set_activation_function('sigmoid')

    neural_net = NeuralNetwork()
    neural_net.define_loss('mean_euclidean')
    neural_net.add_input_layer(input_layer)
    neural_net.add_hidden_layer(hidden_layer)
    #neural_net.add_hidden_layer(hidden_layer2)
    neural_net.add_output_layer(output_layer)

    neural_net.forward_propagation(x)
    print('Debug:\thlayer1.net', hidden_layer.net.shape)
    #print('Debug:\tlayer2.net', hidden_layer2.net.shape)

    neural_net.train_network(x, target_values, 100, 10, 'mean_euclidean', 0.5)


__main__()
