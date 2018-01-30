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
from trainer import *
import os
import re


def __main__():


    filename = 'ML-CUP17-TR.csv'
    x, target_values = ML_CUP_Dataset.load_ML_dataset(filename)

    #x = x[:, 0:5]
    #target_values = target_values[:, 0:5]

    unit_lay=[10,10,10,2]
    af = [ 'relu', 'relu', 'relu', 'linear']
    neural_net = NeuralNetwork.create_advanced_net(4, unit_lay, af, "xavier")
    #neural_net = NeuralNetwork.create_network(5, 10, 3, 2, 'relu', slope=1)

    train_par = {
        'eta': 0.01,
        'alfa': 0.9,
        'lambd': 0.01,
        'epochs': 50,
        'threshold': 0.0,
        'loss': 'mean_euclidean'
    }
    trainer = NeuralTrainer(neural_net, **train_par)
    '''inp, tar = trainer.shuffles(x, target_values)
    trainer._train_no_test(inp, tar, save=True)'''
    trainer._train_no_test(x, target_values, save=True)
    #   trainer.train_rprop_no_test(x, target_values)

    print("output", neural_net.output_layer.output)


'''
    monk_datas = MonkDataset.load_encode_monk('../datasets/monks-3.train')
    monk_targets = monk_datas[0]
    monk_input = monk_datas[1]

    monk_datas_ts = MonkDataset.load_encode_monk('../datasets/monks-3.test')
    monk_targets_ts = monk_datas_ts[0]
    monk_input_ts = monk_datas_ts[1]

    neural_net = NeuralNetwork.create_network(3, 17, 10, 1, 'sigmoid', slope=1)

    train_par = {
        'eta': 0.1,
        'alfa': 0.9,
        'lambd': 0.01,
        'epochs': 100,
        'threshold': 0.0,
        'loss': 'mean_squared_err'
    }
    trainer = NeuralTrainer(neural_net, **train_par)


    #trainer.train_rprop_no_test(monk_input, monk_targets)
    trainer.train_rprop(monk_input, monk_targets, monk_input_ts, monk_targets_ts)
    #trainer.train_network(monk_input,monk_targets, monk_input_ts, monk_targets_ts,"/prova", save=True)
'''

__main__()
