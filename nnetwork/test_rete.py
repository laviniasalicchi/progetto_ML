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
    target_values = np.concatenate((target_x, target_y))
    ''' ---- parte per importare il dataset esterno ---- '''

    '''input_layer = InputLayer(x.shape)
    input_layer.create_weights(x.shape)  # questo è da cambiare perchè probabilmente non funziona con una matrice
    #print(input_layer.weights)
    input = input_layer.net_function(x)

    print(input_layer.output)                       # restituisce []
    input_out = input_layer.layer_output()
    print(input_layer.layer_output())               # restituisce matrice con righe = pattern / colonne = feature
    print(input_layer.n_units[1])

    print("--------------------- HIDDEN LAYER --------------------------")
    hidden_layer = HiddenLayer(input_layer.n_units[1])  #   creazione hidden layer con n unità = n feature
    hidden_layer.create_weights(x.shape[1]+1)             #   creazione matrice pesi: righe = n feature + 1 (per bias)  colonne = n hidden units
    hidden_net = hidden_layer.net_function(input_out)  # net hidden layer: ha come input l'output dell'input layer
    print("HIDDEN NET: ", hidden_net.shape)
    hidden_layer.set_activation_function('tanh')            #   set dell'activation function
    hidden_out = hidden_layer.layer_output()                # output dell'hidden layer
    print("---- HIDDEN OUTPUT ----{", hidden_out.shape,"}\n", hidden_out)

    print("------------------- OUTPUT LAYER --------------------------")
    output_layer = OutputLayer(2)
    output_layer.create_weights(x.shape[1]+1)
    output_net = output_layer.net_function(hidden_out)
    output_layer.set_activation_function('tanh')
    output_out = output_layer.layer_output()
    print("---- OUTPUT OUTPUT ----{", output_out.shape, "}\n", output_out)

    print("**************** N E T W O R K ****************")'''

    input_layer = InputLayer(x.shape)
    input_layer.create_weights(x.shape)

    hidden_layer = HiddenLayer(input_layer.n_units[1])
    hidden_layer.create_weights(x.shape[1] + 1)
    hidden_layer.set_activation_function('tanh')

    hidden_layer2 = HiddenLayer(input_layer.n_units[1])
    hidden_layer2.create_weights(x.shape[1] + 1)
    hidden_layer2.set_activation_function('tanh')

    output_layer = OutputLayer(2)
    output_layer.create_weights(x.shape[1] + 1)
    output_layer.set_activation_function('tanh')

    neural_net = NeuralNetwork()
    neural_net.define_loss('mean_euclidean')
    neural_net.add_input_layer(input_layer)
    neural_net.add_hidden_layer(hidden_layer)
    neural_net.add_hidden_layer(hidden_layer2)
    neural_net.add_output_layer(output_layer)

    print(neural_net.forward_propagation(x))
    neural_net.train_network(x, target_values, 100, 10, 'mean_euclidean', 0.07)


__main__()
