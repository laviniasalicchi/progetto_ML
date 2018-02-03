# -*- coding: utf-8 -*-

# ==============================================================================
# Esegue 5 kfold con k=10 sul modello scelto
# TR mean err = errore medio (media degli errori medi delle kfold)
# TR std err = deviazioni standard degli errori medi
# VL mean err = come sopra ma per il validation set
# TR std err = come sopra ma per il validation set
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================


import random
import time
from monk_dataset import *
from ML_CUP_dataset import ML_CUP_Dataset
from neural_net import NeuralNetwork
from cross_validator import CrossValidator
from trainer import NeuralTrainer
import sys
import datetime as date
import os
from plotter import Plotter

def __main__():

    filename = 'ML-CUP17-TR.csv'
    x = ML_CUP_Dataset.load_ML_dataset(filename)[0]
    target_values = ML_CUP_Dataset.load_ML_dataset(filename)[1]

    # development set
    tr_input = x[:, 0:712]
    tr_target = target_values[:, 0:712]


    activ_funcs = ['tanh', 'tanh','tanh','linear']
    units = [10, 23, 23, 2]

    trainer_param = {
        'eta': 0.1,
        'alfa': 0.9,
        'lambd': 0.01,
        'epochs': 1000  ,
        'threshold': 0.0,
        'loss': 'mean_euclidean'
    }
    net = NeuralNetwork.create_advanced_net(4, units, activ_funcs, 'xavier')

    trainer = NeuralTrainer(net, **trainer_param)
    cross_validator = CrossValidator(trainer)

    kfold_tr_mean_list =  [] # lista di tr medi  per epochs ogni elemento è la media degli errori per epochs del kfold
    kfold_vl_mean_list =  [] # lista di vl medi per epochs

    kfold_means_tr = [] # contiene i valori medi di err del kfold sul tr (singolo valore)
    kfold_means_vl = [] # contiene i valori medi di err del kfold sul tr (singolo valore)


    for i in range(0, 5):
        k_rs = cross_validator.k_fold(tr_input, tr_target, 10)
        tr_err_hist_fold = k_rs['tr_folds_err_h']
        vl_err_hist_fold = k_rs['vl_folds_err_h'] # lista di liste

        kfold_tr_hist_mean = np.mean(np.array(tr_err_hist_fold), axis=0) # per epoch
        kfold_vl_hist_mean = np.mean(np.array(vl_err_hist_fold), axis=0) # per epoch
        kfold_tr_mean_list.append(kfold_tr_hist_mean)
        kfold_vl_mean_list.append(kfold_vl_hist_mean)
        kfold_means_tr.append(k_rs['tr_mean_err'])
        kfold_means_vl.append(k_rs['vl_mean_err'])

    res_to_plot = {
        'tr_folds_err_h': kfold_tr_mean_list,
        'vl_folds_err_h': kfold_vl_mean_list
        }

    final_mean_err_tr = np.mean(kfold_means_tr)
    final_std_err_tr = np.std(kfold_means_tr)

    final_mean_err_vl = np.mean(kfold_means_vl)
    final_std_err_vl= np.std(kfold_means_vl)
    print("TR mean err:", final_mean_err_tr)
    print("TR std err:", final_std_err_tr)
    print("VL mean err:", final_mean_err_vl)
    print("VL std err:", final_std_err_vl)

    Plotter.plot_kfold(res_to_plot)

__main__()
