# -*- coding: utf-8 -*-

import numpy as np

from layer import Layer
from input_layer import InputLayer
from hidden_layer import HiddenLayer
from output_layer import OutputLayer
from neural_net import NeuralNetwork
from cross_validation import kfold_cv
from monk_dataset import Monk_Dataset
from ML_CUP_dataset import ML_CUP_Dataset


def __main__():

    filename = 'ML-CUP17-TR.csv'
    x = ML_CUP_Dataset.load_ML_dataset(filename)[0]
    target_values = ML_CUP_Dataset.load_ML_dataset(filename)[1]

    print(target_values.shape)

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

    output_layer = OutputLayer(1)
    output_layer.create_weights(hidden_layer.n_units)
    output_layer.set_activation_function('sigmoid')

    neural_net = NeuralNetwork()
    neural_net.define_loss('mean_euclidean')
    neural_net.add_input_layer(input_layer)
    neural_net.add_hidden_layer(hidden_layer)
    #neural_net.add_hidden_layer(hidden_layer2)
    neural_net.add_output_layer(output_layer)

    #neural_net.forward_propagation(x)
    #print('Debug:\thlayer1.net', hidden_layer.net.shape)
    #print('Debug:\tlayer2.net', hidden_layer2.net.shape)

    #neural_net.train_network(x, target_values, 100, 10, 'mean_euclidean', 0.5)

    kfold_cv(x, target_values, 5, 10, 'mean_euclidean', 0.5)


    # to do: scommentare la chiamata di train_network

__main__()
