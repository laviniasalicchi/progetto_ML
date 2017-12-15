# -*- coding: utf-8 -*-

import numpy as np
#from classi import NeuronUnit
#import classi.OutputLayer
from layer import Layer



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

    neural_net = NeuralNet()
    input_layer = InputLayer(x.shape)
    input_layer.create_weights(x.shape)  # questo è da cambiare perchè probabilmente non funziona con una matrice
    neural_net.add_input_layer(input_layer)
    hidden_layer = HiddenLayer(input_layer.n_units)
    neural_net.add_hidden_layer(hidden_layer)
    output_layer = OutputLayer(2)
    neural_net.add_output_layer(output_layer)



    __main__()
