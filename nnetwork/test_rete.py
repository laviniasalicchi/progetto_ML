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

    for i in range(0, len(data[:, 0])):
        #print("** ",i," **")
        k = 0
        for j in range(1,11):
            #print(j," - ", data[i][j])
            x[i][k] = data[i][j]
            k = k+1
        target_x[i][0] = data[i][11]
        target_y[i][0] = data[i][12]
    ''' ---- parte per importare il dataset esterno ---- '''

    input_layer = InputLayer(x.shape)
    input_layer.create_weights(x.shape)  # questo è da cambiare perchè probabilmente non funziona con una matrice
    #print(input_layer.weights)
    input = input_layer.net_function(x)

    print(input_layer.output)                       # restituisce []
    inputOut = input_layer.layer_output()
    print(input_layer.layer_output())               # restituisce matrice con righe = pattern / colonne = feature

    print("-----------------------------------------------")
    hidden_layer = HiddenLayer(input_layer.n_units)
    print(hidden_layer.n_units)
    print(inputOut.shape)
    hidden_layer.create_weights(inputOut.shape)
    hids = hidden_layer.net_function(input_layer)
    #print(hids)

    output_layer = OutputLayer(2)

    neural_net = NeuralNetwork()
    neural_net.define_loss('mean_euclidean')
    neural_net.add_input_layer(input_layer)
    neural_net.add_hidden_layer(hidden_layer)
    neural_net.add_output_layer(output_layer)

__main__()
