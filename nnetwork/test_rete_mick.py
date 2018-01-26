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

def k_fold(trainer, input_vect, target_vect, k=4):



    input_size = input_vect.shape[1]
    resto = input_size % k
    fold_size = int(input_size / k)
    start_idx = 0
    acc_list = []
    err_list = []

    print(fold_size)
    print(resto)

    for index in range(1, k + 1):
        if resto != 0:
            end_idx = start_idx + (fold_size + 1) # uso il resto come contatore dei fold che devono avere un elemento in più
            resto = resto - 1
        else:
            end_idx = start_idx + fold_size

        test_kfold = input_vect[:, start_idx:end_idx]
        test_targets = target_vect[:, start_idx:end_idx]

        train_kfold = np.delete(input_vect, np.s_[start_idx:end_idx], axis=1)
        train_targets = np.delete(target_vect, np.s_[start_idx:end_idx], axis=1)

        start_idx = end_idx

        trainer._train_no_test(train_kfold, train_targets)
        test_res = trainer.net.test_network(test_kfold, test_targets)


        err_list.append(test_res[0])
        acc_list.append(test_res[1])

    acc_mean = np.mean(acc_list)
    err_mean = np.mean(err_list)
    # TODO remove printing
    print(acc_list)
    print(acc_mean)
    print(err_list)
    return acc_mean



def __main__():

    """logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)"""

    unit_lay = [17, 5, 5, 5, 1]
    af = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']
    #af = ['tanh', 'tanh', 'tanh', 'tanh', 'tanh']
    #neural_net = NeuralNetwork.create_advanced_net(5, unit_lay, af, "xavier")

    neural_net = NeuralNetwork.create_network(3, 17, 10, 1, 'sigmoid', slope=1)
    args = {
        "eta": 0.1,
        "epochs": 100
        }
    trainer = NeuralTrainer(neural_net, **args)

    monk_datas = MonkDataset.load_encode_monk('/Users/mick/Dati/Università/Pisa/Machine_learning/Prj_info/Progetto_ml/progetto_ML/datasets/monks-3.train')
    monk_targets = monk_datas[0]
    monk_input = monk_datas[1]

    monk_test = MonkDataset.load_encode_monk('/Users/mick/Dati/Università/Pisa/Machine_learning/Prj_info/Progetto_ml/progetto_ML/datasets/monks-3.test')
    monk_test_target = monk_test[0]
    monk_test_input = monk_test[1]

    trainer._train_no_test(monk_input, monk_targets)
    cross = CrossValidator(trainer)
    cross.k_fold(monk_input, monk_targets)




    #neural_net.train_network(monk_input, monk_targets, monk_test_input, monk_test_target, 500, 0.00, 'mean_squared_err', eta=0.02, alfa=0.0, lambd=0.00, final=True)
    #neural_net.train_rprop(monk_input, monk_targets, monk_test_input, monk_test_target, 50, 0.00, 'mean_squared_err', delt0=0.1, delt_max=90)



    #test_result = neural_net.test_network(monk_test_input, monk_test_target)
    #print(test_result[0], test_result[1])

    #cross_validation.kfold_cv(monk_input, monk_targets, 500, 0.0, 'mean_squared_err', eta=0.3, alfa=0.5, lambd=0.01)


__main__()
