# -*- coding: utf-8 -*-

# ==============================================================================
# QUesta classe utilizza un ThreadPoolExecutor per la ricerca degli
# iperparametri nella rete neurale
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================

from cross_validation import *
import concurrent.futures
import random
import time
from monk_dataset import *





def __main__():



    monk_datas = MonkDataset.load_encode_monk('../datasets/monks-1.train')
    monk_targets = monk_datas[0]
    monk_input = monk_datas[1]
    start_grid_search(monk_input, monk_targets, 500, 0.0, 'mean_squared_err')


def start_grid_search(input_vect, target_vect, epochs, threshold, loss_func):
    etas = [0.01, 0.05, 0.1, 0.3, 0.5]
    alfas = [0.5, 0.7, 0.9]
    lambds = [0.01, 0.04, 0.07, 0.1]
    # creo l'executor a cui mandare i task
    executor = concurrent.futures.ThreadPoolExecutor()
    acc_list = []

    for e in etas:
        for a in alfas:
            for l in lambds:
                print("GridSearch started eta=%f, alfas=%f, lambda=%f", e, a, l)
                res = executor.submit(kfold_task, input_vect, target_vect, epochs, threshold, loss_func, eta=e, alfa=a, lambd=l)
                acc_list.append(res.result())

                #NeuralNetwork.saveModel(0, e, a, l, i, acc)
    executor.shutdown(wait=True)
    print("lista accuracies", acc_list)



def kfold_task(input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd):
    acc = kfold_cv(input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd)
    return acc



__main__()
