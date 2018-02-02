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
from grid_searcher import *
import os

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
        '''monk_datas = MonkDataset.load_encode_monk('../datasets/monks-1.train')
        monk_targets = monk_datas[0]
        monk_input = monk_datas[1]
        monk_datas_ts = MonkDataset.load_encode_monk('../datasets/monks-1.test')
        monk_targets_ts = monk_datas_ts[0]
        monk_input_ts = monk_datas_ts[1]

        mod = hold_out(monk_input, monk_targets, 600, 0.0, 'mean_squared_err')
        retraining(mod, monk_input, monk_targets, monk_input_ts, monk_targets_ts, 600, 0.0, 'mean_squared_err')'''

        filename = 'ML-CUP17-TR.csv'
        x = ML_CUP_Dataset.load_ML_dataset(filename)[0]
        target_values = ML_CUP_Dataset.load_ML_dataset(filename)[1]

        mod = hold_out(x, target_values, 500, 0.0, 'mean_euclidean')
        #retraining_noTS(mod, x, target_values)


def hold_out(input_vector, target_value, epochs, threshold, loss_func):
    bound = int(np.rint((input_vector.shape[1]/100)*60))
    training_set = input_vector[:, 0:bound]
    target_training = target_value[:, 0:bound]
    valid_set = input_vector[:, bound:input_vector.shape[1]]
    target_valid = target_value[:, bound:input_vector.shape[1]]

    folder = str("models_CUP/RELAZIONE/")

    models = []

    unit_lay = [10, 5, 5, 2]
    af = ['relu', 'relu', 'relu', 'linear']
    neural_net = NeuralNetwork.create_advanced_net(4, unit_lay, af,"no")

    train_par = {
        'eta': 0.1,
        'alfa': 0.7,
        'lambd': 0.01,
        'epochs': 200,
        'threshold': 0.0,
        'loss': 'mean_euclidean'
    }
    trainer = NeuralTrainer(neural_net, **train_par)
    #trainer.train_network(training_set, target_training, valid_set, target_valid, "", save=True)
    trainer.train_rprop(training_set,target_training, valid_set, target_valid)


def hold_out_grid(input_vector, target_value, epochs, threshold, loss_func):
    bound = int(np.rint((input_vector.shape[1]/100)*60))
    training_set = input_vector[:, 0:bound]
    target_training = target_value[:, 0:bound]
    valid_set = input_vector[:, bound:input_vector.shape[1]]
    target_valid = target_value[:, bound:input_vector.shape[1]]

    grid_layers =[[10,5,2],[10,10,2],[10,5,5,2]]
    #af = ['relu','relu', 'relu', 'linear']
    af = ['relu']

    etas = [0.01, 0.05, 0.1]
    alfas = [0.7, 0.9]
    models = []
    i = 0
    for unit_lay in grid_layers:
        for e in etas:
            for a in alfas:
                folder = str("models_CUP/RELAZIONE/"+str(i))
                train_par = {
                    'eta': e,
                    'alfa': a,
                    'lambd': 0.01,
                    'epochs': 200,
                    'threshold': 0.0,
                    'loss': 'mean_euclidean'
                }
                file = folder+"info.txt"
                print(i,")""eta", e,"alfa", a,"lambda", 0.01, "arr layers", unit_lay)
                with open(file, mode='w') as infomodel:
                    inf = str("eta:"+str(e)+" - alfa:"+str(a)+" - lambda: 0.01")
                    infomodel.write('%s\n' % inf)
                tot_lay = len(unit_lay)
                ac_f = af * (len(unit_lay) - 1)
                ac_f.append('linear')
                neural_net = NeuralNetwork.create_advanced_net(tot_lay, unit_lay, ac_f, "no")
                trainer = NeuralTrainer(neural_net, **train_par)
                trainer.train_network(training_set, target_training, valid_set, target_valid, folder, save=True)
                i=i+1

    return models


__main__()