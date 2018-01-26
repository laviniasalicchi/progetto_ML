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
from neural_net import NeuralNetwork
from cross_validator import CrossValidator
from trainer import NeuralTrainer

""""
class GridSearcher:


    def __init__(self, **kwargs):
        self.etas = kwargs.get('etas', [0.01, 0.05, 0.1, 0.3, 0.5])
        self.alfas = kwargs.get('alfas', [0.5, 0.7, 0.9])
        self.lambds = kwargs.get('lambds', [0.01, 0.04, 0.07, 0.1])
        self.tot_lay = kwargs.get('tot_lay', [3, 4, 5])
        self.n_hid = kwargs.get('n_hid', [5, 10, 15])
        self.act_func = kwargs.get('act_func', ['sigmoid,', 'tanh'])
"""



def __main__():
    if __name__ == '__main__':
        monk_datas = MonkDataset.load_encode_monk('../datasets/monks-1.train')
        monk_targets = monk_datas[0]
        monk_input = monk_datas[1]
        monk_datas_ts = MonkDataset.load_encode_monk('../datasets/monks-1.test')
        monk_targets_ts = monk_datas_ts[0]
        monk_input_ts = monk_datas_ts[1]

        params = {
            'loss': 'mean_euclidean',
            'etas': [0.01, 0.05, 0.1, 0.3, 0.5],
            'alfas': [0.5, 0.7, 0.9],
            'lambds': [0.01, 0.04, 0.07, 0.1],
            'tot_lay': [3, 4, 5],
            'n_hid': [5, 10, 15],
            'act_func': ['sigmoid,', 'tanh']
        }

        input()
        start = time.time() * 1000  # benchmark
        #res = start_adv_grid_search(monk_input, monk_targets, 600, 0.0, 'mean_squared_err')
        mod = grid_search(monk_input, monk_targets, 600, params)
        retraining(mod, monk_input, monk_targets, monk_input_ts, monk_targets_ts, 600, 0.0, 'mean_squared_err')


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

def grid_search(input_vect, target_vect, epochs, trshld=0.00, k=4, **kwargs):
    units_in = input_vect.shape[0] # shape dell'input_vector
    units_out = input_vect.shape[1]
    loss = kwargs.get('loss','mean_euclidean')
    etas = kwargs.get('etas', [0.01, 0.05, 0.1, 0.3, 0.5])
    alfas = kwargs.get('alfas', [0.5, 0.7, 0.9])
    lambds = kwargs.get('lambds', [0.01, 0.04, 0.07, 0.1])
    n_total_layers = kwargs.get('tot_lay', [3, 4, 5])
    n_hidden_units = kwargs.get('n_hid', [5, 10, 15])
    act_func = kwargs.get('act_func', ['sigmoid,', 'tanh'])

    executor = mp.Pool(processes=20)
    res = {}
    models = []
    tot_iter = len(etas) * len(alfas) * len(lambds) * len(act_func) * len(n_hidden_units) * len(n_total_layers)
    count = 1

    for ntl in n_total_layers:
        for nhu in n_hidden_units:
            for af in act_func:
                for e in etas:
                    for a in alfas:
                        for l in lambds:

                            net = NeuralNetwork.create_network(ntl, units_in, nhu, units_out, af)
                            train_par = {
                                'eta': e,
                                'alfa': a,
                                'lambd': l,
                                'epochs': epochs,
                                'threshold': trshld,
                                'loss': loss
                            }
                            trainer = NeuralTrainer(net, train_par)


                            key = "eta=" + str(e) + " alfa=" + str(a) + " lambda" +str(l) + " ntl=" + str(ntl) + " nhu=" + str(nhu) + " act=" + af + "\t"
                            #res[key] = executor.apply_async(kfold_task, (input_vect, target_vect, epochs, threshold, loss_func, e, a, l))
                            res[key] = executor.apply_async(kfold_task,
                             (trainer, input_vect, target_vect, k))
                            acc = res[key].get()

                            models.append({'id': 0, 'accuracy': acc, 'ntl': ntl, 'nhu': nhu, 'af': af, 'eta': e, 'alfa': a, 'lambda': l})
                            print(count, ' of ', tot_iter, ' ', progress, '\% completed')
                            count = count + 1
    executor.close()
    executor.join()
    return models

def adv_grid_search(input_vect, target_vect, epochs, trshld=0.00, k=4, **kwargs):
    units_in = input_vect.shape[0]  # shape dell'input_vector
    units_out = input_vect.shape[1]

    loss = kwargs.get('loss','mean_euclidean')
    etas = kwargs.get('etas', [0.01, 0.05, 0.1, 0.3, 0.5])
    alfas = kwargs.get('alfas', [0.5, 0.7, 0.9])
    lambds = kwargs.get('lambds', [0.01, 0.04, 0.07, 0.1])
    n_total_layers = kwargs.get('tot_lay', [3, 4, 5])
    act_func = kwargs.get('act_func', ['sigmoid,', 'tanh'])
    init = kwargs.get('init', 'def')
    min_hid = 3
    max_hid = 10
    # creo l'executor a cui mandare i task
    executor = mp.Pool(processes=10)
    res = {}
    models = []
    tot_perms = 1
    for n in n_total_layers:
        tot_perms = tot_perms * _tuple_count(n, min_hid, max_hid)
    tot_iter = len(etas) * len(alfas) * len(lambds) * len(af) * tot_perms
    count = 1

    for ntl in n_total_layers:
        permutations = _tuple_generator(ntl, min_hid, max_hid, units_in, units_out)

        for nhu in permutations:
            for af in act_func:
                activ_func = [af] * ntl
                for e in etas:
                    for a in alfas:
                        for l in lambds:
                            net = NeuralNetwork.create_advanced_net(ntl, nhu, activ_func, init)
                            train_par = {
                                eta: e,
                                alfa: a,
                                lambd: l,
                                epochs: epochs,
                                threshold: trshld,
                                loss: loss
                            }
                            trainer = NeuralTrainer(net, train_par)
                            key = "eta=" + str(e) + " alfa=" + str(a) + " lambda" +str(l) + " ntl=" + str(ntl) + " nhu=" + str(nhu) + " act=" + af + "\t"
                            #res[key] = executor.apply_async(kfold_task, (input_vect, target_vect, epochs, threshold, loss_func, e, a, l))
                            res[key] = executor.apply_async(kfold_task,
                             (trainer, input_vect, target_vect, k))
                            acc = res[key].get()

                            models.append({'id': 0, 'accuracy': acc, 'ntl': ntl, 'nhu': nhu, 'af': af, 'eta': e, 'alfa': a, 'lambda': l})
                            progress = (count / float(tot_iter)) * 100

                            print(count, ' of ', tot_iter, ' ', progress, '\% completed')
                            count = count + 1
    executor.close()
    executor.join()
    return models

def kfold_task(trainer, input_vect, target, k=4):
    cross_validator = CrossValidator(trainer)
    acc = cross_validator.k_fold(input_vect, target, k)
    return acc

def _tuple_generator(size, start, end, input_size, out_size):
    result = []
    values = range(start, end + 1)
    perms = list(itertools.permutations(values, size - 2))  # size -2 perchè aggiungiamo a mano input e output
    for tup in perms:
        tup_l = list(tup)
        new_tup = [input_size] + tup_l + [out_size]
        result.append(new_tup)
    #print(result)
    return result

def _tuple_count(size, start, end):
    values = range(start, end + 1)
    perms = list(itertools.permutations(values, size - 2))
    return len(perms)

__main__()