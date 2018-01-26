# -*- coding: utf-8 -*-

# ==============================================================================
# E' una hidden layer
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================

import numpy as np

from layer import Layer
from input_layer import InputLayer
from hidden_layer import HiddenLayer
from output_layer import OutputLayer
from neural_net import NeuralNetwork
from monk_dataset import MonkDataset
from ML_CUP_dataset import ML_CUP_Dataset

class CrossValidator:

    def __init__(self, neural_trainer):
        self.trainer = neural_trainer

        """
        train = trainer(kwargs)
        trainer.train_net(input, tar, test,targ,false)
        """


    def k_fold(self, input_vect, target_vect, k=4):

        trainer = self.trainer

        input_size = input_vect.shape[1]
        resto = input_size % k
        fold_size = int(input_size / k)
        start_idx = 0
        acc_list = []
        err_list = []

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
