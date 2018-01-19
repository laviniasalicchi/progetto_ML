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





def __main__():



    monk_datas = MonkDataset.load_encode_monk('../datasets/monks-1.train')
    monk_targets = monk_datas[0]
    monk_input = monk_datas[1]
    start = time.time() * 1000
    acc_list = mp.Queue()
    start_grid_search(monk_input, monk_targets, 1000, 0.0, 'mean_squared_err', acc_list)
    #grid_search(monk_input, monk_targets, 1000, 0.0, 'mean_squared_err')

    end = time.time() * 1000
    for i in acc_list:
        print(i.get())  
    #kfold_cv_mick(monk_input, monk_targets, 10000, 0.0, 'mean_squared_err', 0, 0, 0)
    print(end-start)


def start_grid_search(input_vect, target_vect, epochs, threshold, loss_func,queue):
    etas = [0.01, 0.05, 0.1, 0.3, 0.5]
    alfas = [0.5, 0.7, 0.9]
    lambds = [0.01, 0.04, 0.07, 0.1]
    # creo l'executor a cui mandare i task
    executor = concurrent.futures.ThreadPoolExecutor()
    executor2 = mp.Pool(processes=mp.cpu_count())
    #acc_list = mp.Queue()

    for e in etas:
        for a in alfas:
            for l in lambds:
                print("GridSearch started eta=%f, alfas=%f, lambda=%f", e, a, l)
                res = executor2.apply_async(kfold_task, (input_vect, target_vect, epochs, threshold, loss_func, e, a, l))
                #res = executor.submit(kfold_task, input_vect, target_vect, epochs, threshold, loss_func, eta=e, alfa=a, lambd=l)
                #acc_list.append(res.result())
                queue.put(res)


    #executor.shutdown(wait=True)
    #return acc_list
    #print(acc_list)



def kfold_task(input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd):
    acc = kfold_cv_mick(input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd)
    return acc


__main__()
