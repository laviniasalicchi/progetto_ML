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
from parallel_grid_search import *

'''
    HOLD-OUT VALIDATION
        1) dividere il dataset in 
            training set
            validation set
        2) per ogni combo di iperparametri:
            - training su TR
            - "test" su validation
        3) prendo la combo con accuracy più alta
        
        4) retraining su TR+VL
        
        5) valutare modello su un TS esterno
'''
def __main__():
    if __name__ == '__main__':
        monk_datas = MonkDataset.load_encode_monk('../datasets/monks-1.train')
        monk_targets = monk_datas[0]
        monk_input = monk_datas[1]
        monk_datas_ts = MonkDataset.load_encode_monk('../datasets/monks-1.test')
        monk_targets_ts = monk_datas_ts[0]
        monk_input_ts = monk_datas_ts[1]

        mod = hold_out(monk_input, monk_targets, 600, 0.0, 'mean_squared_err')
        retraining(mod, monk_input, monk_targets, monk_input_ts, monk_targets_ts, 600, 0.0, 'mean_squared_err')


def hold_out(input_vector, target_value, epochs, threshold, loss_func):
    bound = int(np.rint((input_vector.shape[1]/100)*60))
    training_set = input_vector[:, 0:bound]
    target_training = target_value[:, 0:bound]
    valid_set = input_vector[:, bound:input_vector.shape[1]]
    target_valid = target_value[:, bound:input_vector.shape[1]]

    etas = [0.01, 0.05, 0.1, 0.3, 0.5]
    alfas = [0.5, 0.7, 0.9]
    lambds = [0.01, 0.04, 0.07, 0.1]
    n_total_layers = [3, 4, 5]
    n_hidden_units = [5, 10, 15]  # range(5, 20)
    act_func = ['sigmoid', 'tanh']
    models = []
    i = 0
    for ntl in n_total_layers:
        for nhu in n_hidden_units:
            for af in act_func:
                for e in etas:
                    for a in alfas:
                        for l in lambds:
                            neural_net = NeuralNetwork.create_network(ntl, 17, nhu, 1, af, slope=1)
                            trained = neural_net._train_no_test(training_set, target_training, epochs, threshold, loss_func, eta=e, alfa=a, lambd=l)
                            err, acc = neural_net.test_network(valid_set, target_valid)
                            models.append({'id': i, 'accuracy': acc, 'ntl': ntl, 'nhu': nhu, 'af': af, 'eta': e, 'alfa': a, 'lambda': l})
    return models

__main__()