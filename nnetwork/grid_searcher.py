# -*- coding: utf-8 -*-

# ==============================================================================
# Questa classe utilizza un ThreadPoolExecutor per la ricerca degli
# iperparametri nella rete neurale
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================

import random
import time
from monk_dataset import *
from ML_CUP_dataset import ML_CUP_Dataset
import multiprocessing as mp
import operator
import itertools
from neural_net import NeuralNetwork
from cross_validator import CrossValidator
from trainer import NeuralTrainer
import sys
import datetime as date

def __main__():
    if __name__ == '__main__':
        monk_datas = MonkDataset.load_encode_monk('../datasets/monks-1.train')
        monk_targets = monk_datas[0]
        monk_input = monk_datas[1]
        monk_datas_ts = MonkDataset.load_encode_monk('../datasets/monks-1.test')
        monk_targets_ts = monk_datas_ts[0]
        monk_input_ts = monk_datas_ts[1]

        filename = 'ML-CUP17-TR.csv'
        x = ML_CUP_Dataset.load_ML_dataset(filename)[0]
        target_values = ML_CUP_Dataset.load_ML_dataset(filename)[1]

        params = {
            'units_in': 10, # !!
            'units_out': 2, #!!
            'loss': 'mean_euclidean',
            'etas': [0.01, 0.05, 0.1, 0.3, 0.5],
            'alfas': [0.5, 0.7, 0.9],
            'lambds': [0.01, 0.04, 0.07, 0.1],
            'tot_lay': [3, 4, 5],
            'n_hid': list(range(1,25)),
            'epochs': 500,
            'act_func': ['sigmoid']
        }



        input()
        start = time.time() * 1000  # benchmark
        #   mod = grid_search(monk_input, monk_targets, params)
        #   retraining(mod, monk_input, monk_targets, monk_input_ts, monk_targets_ts)


        mod = grid_search(x, target_values, params)
        retraining(mod, x, target_values)

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


def grid_search(input_vect, target_vect, trshld=0.00, k=4, **kwargs):
    now_m = date.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    now = (now_m.rpartition(':')[0]).replace(":", "_")

    units_in = kwargs.get('unit_in', input_vect.shape[0])
    units_out = kwargs.get('unit_out', target_vect.shape[0])
    loss = kwargs.get('loss', 'mean_euclidean')
    etas = kwargs.get('etas', [0.01, 0.05, 0.1, 0.3, 0.5])
    alfas = kwargs.get('alfas', [0.5, 0.7, 0.9])
    lambds = kwargs.get('lambds', [0.01, 0.04, 0.07, 0.1])
    n_total_layers = kwargs.get('tot_lay', [3, 4])
    n_hidden_units = kwargs.get('n_hid', [5, 10, 15])
    act_func = kwargs.get('act_func', ['sigmoid', 'tanh'])
    epochs = kwargs.get('epochs', 150)

    executor = mp.Pool(processes=20)
    res = {}
    models = []
    tot_iter = len(etas) * len(alfas) * len(lambds) * len(act_func) * len(n_hidden_units) * len(n_total_layers)
    count = 1

    print("start grid search")
    for ntl in n_total_layers:
        for nhu in n_hidden_units:
            for af in act_func:
                for e in etas:
                    for a in alfas:
                        for l in lambds:

                            net_topology = {
                                'units_in': units_in,
                                'units_out': units_out,
                                'tot_lay': ntl,
                                'units_hid': nhu,
                                'act_func': af
                            }

                            train_par = {
                                'eta': e,
                                'alfa': a,
                                'lambd': l,
                                'epochs': epochs,
                                'threshold': trshld,
                                'loss': loss
                            }
                            #trainer = NeuralTrainer(net, **train_par)

                            key = "eta=" + str(e) + " alfa=" + str(a) + " lambda" +str(l) + " ntl=" + str(ntl) + " nhu=" + str(nhu) + " act=" + af + "\t"
                            #res[key] = executor.apply_async(kfold_task, (input_vect, target_vect, epochs, threshold, loss_func, e, a, l))
                            res[key] = executor.apply_async(kfold_task,
                             (net_topology, train_par, input_vect, target_vect, k))
                            acc = res[key].get()
                            print("ntl:", ntl, "nhu:", nhu, " - af:", af, "eta:", e, "alfa:", a, "lambda:", l, "ACCURACY:", acc)

                            models.append({'id': 0, 'accuracy': acc,
                                           'ntl': ntl, 'nhu': nhu, 'u_in': units_in, 'u_out': units_out, 'af': af,
                                           'eta': e, 'alfa': a, 'lambda': l,
                                           'epochs': epochs, 'trshld': trshld, 'loss': loss})
                            progress = (count / tot_iter) * 100
                            mess = 'Progress: {} %' + '    (' + str(count) + ' of ' + str(tot_iter) + ')'
                            mess = mess.format(int(progress))
                            print(mess  , end='\r')
                            sys.stdout.flush()
                            if int(progress) > 0 and (int(progress) % 20) == 0:
                                current_par = {
                                    'eta': e,
                                    'alfa': a,
                                    'lambd': l,
                                    'tot_lay': ntl,
                                    'units_hid': nhu,
                                    'act_func': af
                                }
                                _backup_grid_search(models, current_par, now)
                            count = count + 1
    executor.close()
    executor.join()
    return models

def adv_grid_search(input_vect, target_vect, trshld=0.00, k=4, **kwargs):
    now = date.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')

    units_in = kwargs.get('unit_in', input_vect.shape[0])
    units_out = kwargs.get('unit_out', target_vect.shape[0])
    epochs = kwargs.get('epochs', 150)
    loss = kwargs.get('loss', 'mean_squared_err')
    etas = kwargs.get('etas', [0.01, 0.05, 0.1, 0.3, 0.5])
    alfas = kwargs.get('alfas', [0.5, 0.7, 0.9])
    lambds = kwargs.get('lambds', [0.01, 0.04, 0.07, 0.1])
    n_total_layers = kwargs.get('tot_lay', [3, 4, 5])
    act_func = kwargs.get('act_func', ['sigmoid', 'tanh'])
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
    tot_iter = len(etas) * len(alfas) * len(lambds) * len(act_func) * tot_perms
    count = 1

    for ntl in n_total_layers:
        permutations = _tuple_generator(ntl, min_hid, max_hid, units_in, units_out)

        for nhu in permutations:
            for af in act_func:
                activ_func = [af] * ntl
                for e in etas:
                    for a in alfas:
                        for l in lambds:

                            net_topology = {
                                'un_lays': nhu,
                                'units_out': units_out,
                                'tot_lay': ntl,
                                'init': init,
                                'act_func': af
                            }
                            train_par = {
                                'eta': e,
                                'alfa': a,
                                'lambd': l,
                                'epochs': epochs,
                                'threshold': trshld,
                                'loss': loss
                            }
                            key = "eta=" + str(e) + " alfa=" + str(a) + " lambda" +str(l) + " ntl=" + str(ntl) + " nhu=" + str(nhu) + " act=" + af + "\t"
                            #res[key] = executor.apply_async(kfold_task, (input_vect, target_vect, epochs, threshold, loss_func, e, a, l))
                            res[key] = executor.apply_async(kfold_adv_task,
                             (net_topology, train_par, input_vect, target_vect, k))
                            acc = res[key].get()

                            models.append({'id': 0, 'accuracy': acc,
                                           'ntl': ntl, 'nhu': nhu, 'u_in': units_in, 'u_out': units_out, 'af': af,
                                           'eta': e, 'alfa': a, 'lambda': l,
                                           'epochs': epochs, 'trshld': trshld, 'loss': loss})
                            progress = (count / tot_iter) * 100
                            mess = 'Progress: {} %' + '    (' + str(count) + ' of ' + str(tot_iter) + ')'
                            mess = mess.format(int(progress))
                            print(mess, end='\r')
                            sys.stdout.flush()
                            if int(progress) > 0 and (int(progress) % 20) == 0:
                                current_par = {
                                    'eta': e,
                                    'alfa': a,
                                    'lambd': l,
                                    'un_lays': nhu,
                                    'units_out': units_out,
                                    'tot_lay': ntl,
                                    'init': init,
                                    'act_func': af
                                }
                                _backup_grid_search(models, current_par, now)
                            count = count + 1
    executor.close()
    executor.join()
    return models


def kfold_adv_task(net_topology, trainer_param, input_vect, target, k=4):
    acc = CrossValidator.kfold_grid_adv(net_topology, trainer_param, input_vect, target, k=4)
    return acc

def kfold_task(net_topology, trainer_param, input_vect, target, k=4):

    acc = CrossValidator.k_fold_grid(net_topology, trainer_param, input_vect, target, k=4)
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


def _backup_grid_search(models, params, now):
    """
    Salva gli iperparametri dei modelli testati.
    Copia la lista dei modelli e la ordina salvano i migliori 5 trovati.
    in un file separato salva l'ultimo valore degli iperparametri testato
    """
    modelli = sorted(list(models), key=lambda k: k['accuracy'], reverse=True)
    filename = 'grid_sear_' + now + '.txt'
    filename2 = 'last_params_' + now + '.txt'
    with open(filename, mode='w') as models_backup:
        for i in range(1, 6):
            mod = str(str(modelli[i]))
            models_backup.write('%s\n' % mod)
    with open(filename2, mode='w') as par_backup:
        par = str(params)
        par_backup.write('%s\n' % par)




    #for dict in modelli:


def retraining(models, input_vect, target_vect, input_test, target_test):
    models = sorted(models, key=lambda k: k['accuracy'], reverse=True)
    print("sorted models", models)
    j = 0
    for m in models:
        ntl = m['ntl']
        nhu = m['nhu']
        u_in = m['u_in']
        u_out = m['u_out']
        af = m['af']
        eta = m['eta']
        alfa = m['alfa']
        lambd = m['lambda']
        epochs = m['epochs']
        threshold = m['trshld']
        loss = m['loss']

        train_par = {
            'eta': eta,
            'alfa': alfa,
            'lambd': lambd,
            'epochs': epochs,
            'threshold': threshold,
            'loss': loss
        }
        print("model: tot layer", ntl, " - hidden units", nhu, " - eta", eta, " - alfa", alfa, " - lambda", lambd, " - activ func", af)
        net = NeuralNetwork.create_network(ntl, u_in, nhu, u_out, af, slope=1)
        trainer = NeuralTrainer(net, **train_par)

        n_mod = str(j)
        folder = str("models/finals/Model" +n_mod+ "/")
        trainer._train_no_test(input_vect, target_vect, save=True)
        #   DECOMM  weights, error = trainer.train_network(input_vect, target_vect, input_test, target_test, folder, save=True)
        print("in retraining - fine train_network")

        print("model: tot layer", ntl," - hidden units",nhu," - eta", eta, " - alfa", alfa," - lambda", lambd, " - activ func", af  )

        #   DEOMM   net.saveModel(weights, eta, alfa, lambd, ntl, nhu, af, u_in, u_out, epochs, threshold, loss, j, final=True)
        j += 1
        if j == 5:
            break

def grid_search_groups(input_vect, target_vect, g, trshld=0.00, k=4, **kwargs):
    now = date.datetime.now().strftime('%d-%m-%Y_%H_%M_%S')

    units_in = kwargs.get('unit_in', input_vect.shape[0])
    units_out = kwargs.get('unit_out', target_vect.shape[0])

    if(g==1):
        # gruppo 1
        print("in gruppo 1")
        loss = kwargs.get('loss', 'mean_squared_err')
        etas = kwargs.get('etas', [0.01, 0.03])
        alfas = kwargs.get('alfas', [0.5, 0.7, 0.9])
        lambds = kwargs.get('lambds', [0.01, 0.04])
        n_total_layers = kwargs.get('tot_lay', [3, 4])
        n_hidden_units = kwargs.get('n_hid', [5, 10])
        act_func = kwargs.get('act_func', ['sigmoid'])
        epochs = kwargs.get('epochs', 200)

    elif(g==2):
        # gruppo 2
        print("in gruppo 2")
        loss = kwargs.get('loss', 'mean_squared_err')
        etas = kwargs.get('etas', [0.05, 0.07, 0.1])
        alfas = kwargs.get('alfas', [0.7, 0.9])
        lambds = kwargs.get('lambds', [0.01, 0.04, 0.07])
        n_total_layers = kwargs.get('tot_lay', [3, 4])
        n_hidden_units = kwargs.get('n_hid', [5, 10])
        act_func = kwargs.get('act_func', ['sigmoid'])
        epochs = kwargs.get('epochs', 200)

    elif(g==3):
        # gruppo 3
        print("in gruppo 3")
        loss = kwargs.get('loss', 'mean_squared_err')
        etas = kwargs.get('etas', [0.3, 0.5])
        alfas = kwargs.get('alfas', [0.7, 0.9])
        lambds = kwargs.get('lambds', [0.01, 0.04, 0.07])
        n_total_layers = kwargs.get('tot_lay', [3, 4])
        n_hidden_units = kwargs.get('n_hid', [5, 10])
        act_func = kwargs.get('act_func', ['sigmoid'])
        epochs = kwargs.get('epochs', 200)

    executor = mp.Pool(processes=20)
    res = {}
    models = []
    tot_iter = len(etas) * len(alfas) * len(lambds) * len(act_func) * len(n_hidden_units) * len(n_total_layers)
    count = 1

    print("start grid search")
    for ntl in n_total_layers:
        for nhu in n_hidden_units:
            for af in act_func:
                for e in etas:
                    for a in alfas:
                        for l in lambds:

                            net_topology = {
                                'units_in': units_in,
                                'units_out': units_out,
                                'tot_lay': ntl,
                                'units_hid': nhu,
                                'act_func': af
                            }

                            train_par = {
                                'eta': e,
                                'alfa': a,
                                'lambd': l,
                                'epochs': epochs,
                                'threshold': trshld,
                                'loss': loss
                            }
                            #trainer = NeuralTrainer(net, **train_par)

                            key = "eta=" + str(e) + " alfa=" + str(a) + " lambda" +str(l) + " ntl=" + str(ntl) + " nhu=" + str(nhu) + " act=" + af + "\t"
                            #res[key] = executor.apply_async(kfold_task, (input_vect, target_vect, epochs, threshold, loss_func, e, a, l))
                            acc_tmp = []
                            # 3 kfold e acc = media dei 3
                            for rep in range(3):
                                res[key] = executor.apply_async(kfold_task,(net_topology, train_par, input_vect, target_vect, k))
                                acc = res[key].get()
                                acc_tmp = np.append(acc_tmp, acc)
                            acc = np.mean(acc_tmp)
                            print("ntl:", ntl, "nhu:", nhu, " - af:", af, "eta:", e, "alfa:", a, "lambda:", l, "ACCURACY:", acc)


                            models.append({'id': 0, 'accuracy': acc,
                                           'ntl': ntl, 'nhu': nhu, 'u_in': units_in, 'u_out': units_out, 'af': af,
                                           'eta': e, 'alfa': a, 'lambda': l,
                                           'epochs': epochs, 'trshld': trshld, 'loss': loss})
                            progress = (count / tot_iter) * 100
                            mess = 'Progress: {} %' + '    (' + str(count) + ' of ' + str(tot_iter) + ')'
                            mess = mess.format(int(progress))
                            print(mess  , end='\r')
                            sys.stdout.flush()
                            if int(progress) > 0 and (int(progress) % 20) == 0:
                                current_par = {
                                    'eta': e,
                                    'alfa': a,
                                    'lambd': l,
                                    'tot_lay': ntl,
                                    'units_hid': nhu,
                                    'act_func': af
                                }
                                _backup_grid_search(models, current_par, now)
                            count = count + 1
    executor.close()
    executor.join()
    return models

__main__()
