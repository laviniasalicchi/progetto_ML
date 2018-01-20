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
import multiprocessing as mp
import parallelTestModule


def __main__():
    if __name__ == '__main__':




        monk_datas = MonkDataset.load_encode_monk('../datasets/monks-1.train')
        monk_targets = monk_datas[0]
        monk_input = monk_datas[1]
        start = time.time() * 1000  # benchmark
        res = start_grid_search(monk_input, monk_targets, 100, 0.0, 'mean_squared_err')

        #grid_search(monk_input, monk_targets, 1000, 0.0, 'mean_squared_err')

        end = time.time() * 1000
        for key, value in res.items():
            print(key, value.get())
        #kfold_cv_mick(monk_input, monk_targets, 10000, 0.0, 'mean_squared_err', 0, 0, 0)
        input()
        print("TIME: ", end-start)



def start_grid_search(input_vect, target_vect, epochs, threshold, loss_func):
    etas = [0.01, 0.05, 0.1, 0.3, 0.5]
    alfas = [0.5, 0.7, 0.9]
    lambds = [0.01, 0.04, 0.07, 0.1]
    n_total_layers = [3, 4, 5]
    n_hidden_units = range(5, 20)
    act_func = ['sigmoid', 'tanh']
    # creo l'executor a cui mandare i task
    executor = mp.Pool(processes=20)
    res = {}

    for ntl in n_total_layers:
        for nhu in n_hidden_units:
            for af in act_func:
                for e in etas:
                    for a in alfas:
                        for l in lambds:
                            print("GridSearch started eta=%f, alfas=%f, lambda=%f", e, a, l)
                            key = "eta=" + str(e) + " alfa=" + str(a) + " lambda" +str(l) + " ntl=" + str(ntl) + "nhu=" + str(nhu) + "act=" + af + "\t"
                            #res[key] = executor.apply_async(kfold_task, (input_vect, target_vect, epochs, threshold, loss_func, e, a, l))
                            res[key] = executor.apply_async(kfold_task_net_topology,
                             (input_vect, target_vect, epochs, threshold, loss_func, e, a, l, ntl, nhu, af))



    executor.close()
    executor.join()
    return res




def kfold_task(input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd):
    acc = kfold_cv_mick(input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd)
    return acc

def kfold_task_net_topology(input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd, ntl, nhu, act_func):
    acc = kfold_net_topology(input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd, ntl, nhu, act_func)
    return acc



__main__()
