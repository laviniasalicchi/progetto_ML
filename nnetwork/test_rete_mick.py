# -*- coding: utf-8 -*-

import numpy as np
from layer import Layer
from input_layer import InputLayer
from hidden_layer import HiddenLayer
from output_layer import OutputLayer
from neural_net import NeuralNetwork
from monk_dataset import *
from ML_CUP_dataset import ML_CUP_Dataset


def __main__():

    '''
    filename = 'ML-CUP17-TR.csv'
    x = ML_CUP_Dataset.load_ML_dataset(filename)[0]
    target_values = ML_CUP_Dataset.load_ML_dataset(filename)[1]

    input_layer = InputLayer(x.shape[0])
    input_layer.create_weights(x.shape[0])

    hidden_layer = HiddenLayer(5)
    hidden_layer.create_weights(input_layer.n_units)
    hidden_layer.set_activation_function('sigmoid')

    output_layer = OutputLayer(1)
    output_layer.create_weights(hidden_layer.n_units)
    output_layer.set_activation_function('sigmoid')

    neural_net = NeuralNetwork()
    neural_net.define_loss('mean_euclidean')
    neural_net.add_input_layer(input_layer)
    neural_net.add_hidden_layer(hidden_layer)
    neural_net.add_output_layer(output_layer)'''

    neural_net = NeuralNetwork.create_network(3, 17, 2, 1, 'sigmoid')

    monk_datas = MonkDataset.load_encode_monk('/Users/mick/Dati/Università/Pisa/Machine_learning/Prj_info/Progetto_ml/monks-1.train')
    monk_targets = monk_datas[0]
    monk_input = monk_datas[1]
    neural_net.train_network(monk_input, monk_targets, 1000, 0.00, 'mean_squared_err', 0.1)




__main__()
