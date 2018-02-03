# -*- coding: utf-8 -*-

import numpy as np
from layer import Layer
from input_layer import InputLayer
from hidden_layer import HiddenLayer
from output_layer import OutputLayer
from neural_net import NeuralNetwork
from monk_dataset import *
from ML_CUP_dataset import ML_CUP_Dataset
import cross_validation
import logging
from trainer import *
from cross_validator import CrossValidator




def __main__():

    activ_funcs = ['tanh', 'tanh','tanh','linear']
    units = [10, 23, 23, 2]
    net = NeuralNetwork.create_advanced_net(4, units, activ_funcs, 'xavier')

    net.save_net('abbazzia/')





    #neural_net.train_network(monk_input, monk_targets, monk_test_input, monk_test_target, 500, 0.00, 'mean_squared_err', eta=0.02, alfa=0.0, lambd=0.00, final=True)
    #neural_net.train_rprop(monk_input, monk_targets, monk_test_input, monk_test_target, 50, 0.00, 'mean_squared_err', delt0=0.1, delt_max=90)



    #test_result = neural_net.test_network(monk_test_input, monk_test_target)
    #print(test_result[0], test_result[1])

    #cross_validation.kfold_cv(monk_input, monk_targets, 500, 0.0, 'mean_squared_err', eta=0.3, alfa=0.5, lambd=0.01)


__main__()
