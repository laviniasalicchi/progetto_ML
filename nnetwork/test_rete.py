# -*- coding: utf-8 -*-

import numpy as np
from layer import Layer
from input_layer import InputLayer
from hidden_layer import HiddenLayer
from output_layer import OutputLayer
from neural_net import NeuralNetwork
from monk_dataset import *
from ML_CUP_dataset import ML_CUP_Dataset
from cross_validation import *
from holdout_validation import *
import os
import re


def __main__():


    filename = 'ML-CUP17-TR.csv'
    x = ML_CUP_Dataset.load_ML_dataset(filename)[0]
    target_values = ML_CUP_Dataset.load_ML_dataset(filename)[1]

    '''
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

    monk_datas = MonkDataset.load_encode_monk('../datasets/monks-1.train')
    monk_targets = monk_datas[0]
    monk_input = monk_datas[1]

    #kfold_cv(monk_input, monk_targets, 800, 0.00001, 'mean_squared_err', eta=0.3, alfa=0.5, lambd=0.01)

    monk_datas_ts = MonkDataset.load_encode_monk('../datasets/monks-1.test')
    monk_targets_ts = monk_datas_ts[0]
    monk_input_ts = monk_datas_ts[1]

    neural_net = NeuralNetwork.create_network(3, 17, 10, 1, 'sigmoid', slope=1)

    #   neural_net.train_network(monk_input, monk_targets, monk_input_ts, monk_targets_ts, 300, 0.00, 'mean_squared_err', eta=0.1, alfa=0.9, lambd=0.01, final=True)
    neural_net.train_rprop(monk_input, monk_targets, monk_input_ts, monk_targets_ts, 300, 0.00, 'mean_squared_err', 0, 500)
    #   neural_net.train_network(monk_input, monk_targets, 300, 0.00001, 'mean_squared_err', eta=0.4, alfa=0.9, lambd=0.1, final=True)
    #neural_net.test_existing_model(monk_input_ts,monk_targets_ts)

    #hold_out(monk_input, monk_targets, monk_input_ts, monk_targets_ts, 1000, 0.00001, 'mean_squared_err')


    #   grid_search(monk_input, monk_targets, 500, 0.0, 'mean_squared_err')

    #   NeuralNetwork.test_existing_model(monk_input_ts, monk_targets_ts)



    #neural_net_test = NeuralNetwork.create_network(3, 17, 5, 1, 'sigmoid', slope=1)

    '''neural_net_test = NeuralNetwork.create_network(3, 17, 5, 1, 'sigmoid')

    path = "models/finals/"

    neural_net_test.test_existing_model(monk_input, monk_targets, path)'''







__main__()
