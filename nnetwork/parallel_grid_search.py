# -*- coding: utf-8 -*-

# ==============================================================================
# Questa classe utilizza un ThreadPoolExecutor per la ricerca degli
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
import operator
import itertools


def __main__():
    if __name__ == '__main__':
        monk_datas = MonkDataset.load_encode_monk('../datasets/monks-1.train')
        monk_targets = monk_datas[0]
        monk_input = monk_datas[1]
        print(monk_targets.shape)
        print(monk_input.shape)
        input()
        start = time.time() * 1000  # benchmark
        res = start_adv_grid_search(monk_input, monk_targets, 600, 0.0, 'mean_squared_err')

        #grid_search(monk_input, monk_targets, 1000, 0.0, 'mean_squared_err')

        end = time.time() * 1000
        ''''# ottieni i valori
        # ottieni i valori
        for key, value in res.items():
            res[key] = value.get()
        # sorting
        sorted_res = sorted(res.items(), key=operator.itemgetter(1))
        for x in sorted_res:
            print(x[0], x[1])


        #kfold_cv_mick(monk_input, monk_targets, 10000, 0.0, 'mean_squared_err', 0, 0, 0)'''
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
    models=[]
    i=0
    for ntl in n_total_layers:
        for nhu in n_hidden_units:
            for af in act_func:
                for e in etas:
                    for a in alfas:
                        for l in lambds:
                            print("GridSearch started eta=%f, alfas=%f, lambda=%f", e, a, l)
                            key = "eta=" + str(e) + " alfa=" + str(a) + " lambda" +str(l) + " ntl=" + str(ntl) + " nhu=" + str(nhu) + " act=" + af + "\t"
                            #res[key] = executor.apply_async(kfold_task, (input_vect, target_vect, epochs, threshold, loss_func, e, a, l))
                            res[key] = executor.apply_async(kfold_task_net_topology,
                             (input_vect, target_vect, epochs, threshold, loss_func, e, a, l, ntl, nhu, af))
                            acc = res[key].get()
                            models.append({'id': i, 'accuracy': acc, 'ntl': ntl, 'nhu': nhu, 'af': af, 'eta': e, 'alfa': a, 'lambda': l})

    print("fine for")
    models = sorted(models, key=lambda k: k['accuracy'], reverse=True)
    print("sorted models", models)
    j = 0
    for m in models:
        ntl = m['ntl']
        nhu = m['nhu']
        af = m['af']
        eta = m['eta']
        alfa = m['alfa']
        lambd = m['lambda']

        neural_net = NeuralNetwork.create_network(ntl, 17, nhu, 1, af, slope=1)
        weights, error = neural_net.train_network(input_vect, target_vect, epochs, threshold, loss_func, eta, alfa, lambd)  # , final=True)
        accuracy = neural_net.accuracy(neural_net.output_layer.output, target_vect)
        neural_net.saveModel(weights, eta, alfa, lambd, ntl, nhu, af, j, accuracy, final=True)
        j += 1
        if j == 5:
            break

    executor.close()
    executor.join()
    return res

def start_adv_grid_search(input_vect, target_vect, epochs, threshold, loss_func):
    etas = [0.01, 0.05, 0.1, 0.3, 0.5]
    alfas = [0.5, 0.7, 0.9]
    lambds = [0.01, 0.04, 0.07, 0.1]
    n_total_layers = [3, 4, 5]
    min_hid = 3
    max_hid = 15
    act_func = ['sigmoid', 'tanh']
    # creo l'executor a cui mandare i task
    executor = mp.Pool(processes=20)
    res = {}
    models = []
    for ntl in n_total_layers:
        permutations = _tuple_generator(ntl, min_hid, max_hid, 17, 1) #

        for nhu in permutations:
            for af in act_func:
                activ_func = [af] * ntl
                for e in etas:
                    for a in alfas:
                        for l in lambds:
                            print("GridSearch started eta=%f, alfas=%f, lambda=%f", e, a, l)
                            key = "eta=" + str(e) + " alfa=" + str(a) + " lambda" +str(l) + " ntl=" + str(ntl) + " nhu=" + str(nhu) + " act=" + af + "\t"
                            #res[key] = executor.apply_async(kfold_task, (input_vect, target_vect, epochs, threshold, loss_func, e, a, l))
                            res[key] = executor.apply_async(kfold_task_adv_net_topology,
                             (input_vect, target_vect, epochs, threshold, loss_func, e, a, l, ntl, nhu, activ_func))
                            acc = res[key].get()
                            models.append({'id': 1, 'accuracy': acc, 'ntl': ntl, 'nhu': nhu, 'af': activ_func, 'eta': e, 'alfa': a, 'lambda': l})

    print("fine for")
    models = sorted(models, key=lambda k: k['accuracy'], reverse=True)
    print("sorted models", models)
    j = 0
    for m in models:
        ntl = m['ntl']
        nhu = m['nhu']
        af = m['af']
        eta = m['eta']
        alfa = m['alfa']
        lambd = m['lambda']

        neural_net = NeuralNetwork.create_network(ntl, 17, nhu, 1, af, slope=1)
        weights, error = neural_net._train_no_test(input_vect, target_vect, epochs, threshold, loss_func, eta, alfa, lambd)  # , final=True)
        accuracy = neural_net.accuracy(neural_net.output_layer.output, target_vect)
        neural_net.saveModel(weights, eta, alfa, lambd, ntl, nhu, af, j, accuracy, final=True)
        j += 1
        if j == 5:
            break

    executor.close()
    executor.join()
    return res




def kfold_task(input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd):
    acc = kfold_cv_mick(input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd)
    return acc

def kfold_task_net_topology(input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd, ntl, nhu, act_func):
    acc = kfold_net_topology(input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd, ntl, nhu, act_func)
    return acc

def kfold_task_adv_net_topology(input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd, ntl, nhu, act_func):
    acc = kfold_adv_net_topology(input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd, ntl, nhu, act_func)
    return acc

def _tuple_generator(size, start, end, input_size, out_size):
    result = []
    values = range(start, end + 1)
    perms = list(itertools.permutations(values, size - 2))  # size -2 perchè aggiungiamo a mano input e output
    for tup in perms:
        tup_l = list(tup)
        new_tup = [input_size] + tup_l + [out_size]
        result.append(new_tup)
    return result

__main__()
