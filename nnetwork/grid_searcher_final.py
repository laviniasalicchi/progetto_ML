# -*- coding: utf-8 -*-

# ==============================================================================
# Questa classe effettua grid searches
#
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

def generate_unit_per_layer(n_in, n_hid, n_out, tot_lay):
    """
    funzione helper per generare una lista con il numero di unità per
    ogni layer.
    La lista verrà passata tramite dizionario per istanziare la rete
    con la topologia scelta
    """
    res = []
    res = [n_in] + ([n_hid] * (tot_lay - 2)) + [n_out]
    return res

def adv_grid_search_cup(input_vect, target_vect, trshld=0.00, k=4, **kwargs):
    """
    Funzione per effettuare una grid search.

    input_vect = matrice di input
    target_vect = matrice target
    threshold = soglia di errore sotto la quale fermare il training.
                Non implementata al momento
    kwargs = keywords arguments:
                unit_in = Numero unità input layer. per default numero di righe della
                          matrice di input
                unit_out = Numero unità output layer. Per default numero di righe della matrice target
                epochs = numero di epochs di training
                etas =  Lista di valori di learning rate da testare.
                alfas = Lista di valori di momentum da testare
                lambds = Lista di valori di lamba da testare
                tot_lay = Lista di valori per il massimo numero totale  di layer della rete
                act_func = Lista di stringhe. Nomi di funzioni di attivazione da testare per gli hidden layer
                           Vedere classe layer

                init = Stringa. valori possibili: def = inizializzazione random. xavier = fan in
                n_hid = Lista di interi. Numero di unità di hidden layer da testare.
                loss = Stringa. Tipo di loss function della rete. Valori possibili: mean_euclidean, mean_squared_err

    """
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
    unit_hid = kwargs.get('n_hid', [10])

    models = []
    tot_iter = len(etas) * len(alfas) * len(lambds) * len(act_func) * len(unit_hid) * len(n_total_layers)
    count = 1
    current_par = {}

    for ntl in n_total_layers:
        for nhu in unit_hid:
            units_per_lay = generate_unit_per_layer(units_in, nhu, units_out, ntl)
            for af in act_func:
                activ_func = [af] * (ntl - 1)
                activ_func = activ_func + ['linear'] # output is linear
                for e in etas:
                    for a in alfas:
                        for l in lambds:

                            net_topology = {
                                'un_lays': units_per_lay,
                                'units_out': units_out,
                                'tot_lay': ntl,
                                'init': init,
                                'act_func': activ_func
                            }
                            trainer_param = {
                                'eta': e,
                                'alfa': a,
                                'lambd': l,
                                'epochs': epochs,
                                'threshold': trshld,
                                'loss': loss
                            }

                            err = CrossValidator.kfold_grid_adv(net_topology, trainer_param, input_vect, target_vect, k=10)

                            mod = {
                                'id': 0,
                                 'err': err,
                                 'ntl': ntl,
                                 'nhu': nhu,
                                 'u_in': units_in,
                                 'u_out': units_out,
                                 'af': af,
                                 'eta': e,
                                 'alfa': a,
                                 'lambda': l,
                                 'epochs': epochs,
                                 'trshld': trshld,
                                 'loss': loss
                            }

                            models.append(mod)
                            progress = (count / tot_iter) * 100
                            mess = 'Progress: {} %' + '    (' + str(count) + ' of ' + str(tot_iter) + ')'
                            mess = mess.format(int(progress))
                            print(mess, end='\r')
                            if int(progress) > 0 and (int(progress) % 15) == 0:
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
    _backup_grid_search(models, current_par, now)

    return models

def adv_grid_search_monk(input_vect, target_vect, trshld=0.00, k=4, **kwargs):
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
    unit_hid = kwargs.get('n_hid', [10])

    models = []
    tot_iter = len(etas) * len(alfas) * len(lambds) * len(act_func) * len(unit_hid) * len(n_total_layers)
    count = 1
    current_par = {}


    for ntl in n_total_layers:
        for nhu in unit_hid:
            units_per_lay = generate_unit_per_layer(units_in, nhu, units_out, ntl)
            for af in act_func:
                activ_func = [af] * (ntl - 1)
                activ_func = activ_func + ['linear'] # output is linear
                for e in etas:
                    for a in alfas:
                        for l in lambds:

                            net_topology = {
                                'un_lays': units_per_lay,
                                'units_out': units_out,
                                'tot_lay': ntl,
                                'init': init,
                                'act_func': activ_func
                            }
                            trainer_param = {
                                'eta': e,
                                'alfa': a,
                                'lambd': l,
                                'epochs': epochs,
                                'threshold': trshld,
                                'loss': loss
                            }

                            acc = CrossValidator.kfold_grid_adv(net_topology, trainer_param, input_vect, target_vect, k=10)

                            mod = {
                                'id': 0,
                                 'accuracy': acc,
                                 'ntl': ntl,
                                 'nhu': nhu,
                                 'u_in': units_in,
                                 'u_out': units_out,
                                 'af': af,
                                 'eta': e,
                                 'alfa': a,
                                 'lambda': l,
                                 'epochs': epochs,
                                 'trshld': trshld,
                                 'loss': loss
                            }

                            models.append(mod)
                            progress = (count / tot_iter) * 100
                            mess = 'Progress: {} %' + '    (' + str(count) + ' of ' + str(tot_iter) + ')'
                            mess = mess.format(int(progress))
                            print(mess, end='\r')
                            if int(progress) > 0 and (int(progress) % 15) == 0:
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
                                _backup_grid_search(models, current_par, now, False)

                            count = count + 1
    _backup_grid_search(models, current_par, now, False)

    return models

def _backup_grid_search(models, params, now, err=True):
    """
    Salva gli iperparametri dei modelli testati.
    Copia la lista dei modelli e la ordina salvano i migliori 10 trovati.
    in un file separato salva l'ultimo valore degli iperparametri testato
    models = lista di dizionari da salvare
    params = ultimi parametri testati
    now = timestamp di inizio grid search
    err = se True, il sorting è sull'errore dei modelli, altrimenti sull'accuracy
    """
    if err:
        modelli = sorted(list(models), key=lambda k: k['err'], reverse=False)
    else:
        modelli = sorted(list(models), key=lambda k: k['accuracy'], reverse=False)

    mod_count = len(models)
    directory = 'grid_searches'
    path = directory + "/"
    if not os.path.exists(path):
        print(test)
        os.makedirs(path)
    filename = 'grid_sear_' + now + '.txt'
    filename2 = 'last_params_' + now + '.txt'
    with open(os.path.join(directory, filename), mode='w') as models_backup:
        for i in range(0, 11):
            if i <= (mod_count - 1):
                mod = str(str(modelli[i]))
                models_backup.write('%s\n' % mod)
    with open(os.path.join(directory, filename2), mode='w') as par_backup:
        par = str(params)
        par_backup.write('%s\n' % par)




def __main__():


    print('Input 1 for a grid search on Monk1, 2 for a grid search on Cup dataset')
    choice = input()
    if choice == '1':
        monk_datas = MonkDataset.load_encode_monk('../datasets/monks-1.train')
        monk_targets = monk_datas[0]
        monk_input = monk_datas[1]

        # parametri per la grid search
        params = {
            'units_in': 17, # unità di input
            'units_out': 1, # unità di out
            'loss': 'mean_euclidean',
            'etas': [0.1],
            'alfas': [0.9],
            'lambds': [0.01],
            'tot_lay': [4],
            'n_hid': [26],
            'epochs': 800,
            'act_func': ['tanh']
        }
        mods = adv_grid_search_cup(monk_input, monk_targets, **params)
    if choice == '2':
         filename = 'ML-CUP17-TR.csv'
         x = ML_CUP_Dataset.load_ML_dataset(filename)[0]
         target_values = ML_CUP_Dataset.load_ML_dataset(filename)[1]
         # development set
         tr_input = x[:, 0:712]
         tr_target = target_values[:, 0:712]
         # test set
         ts_input = x[:, 712:]
         ts_target = target_values[:, 712:0]

         eta = [0.01, 0.05, 0.1, 0.2]
         alfa = [0.7, 0.9]
         lambds = [0.01, 0.02]

         # parametri per la grid search
         params = {
             'units_in': 10, # unità di input
             'units_out': 2, # unità di out
             'loss': 'mean_euclidean',
             'etas': [0.1],
             'alfas': [0.9],
             'lambds': [0.01],
             'tot_lay': [4],
             'n_hid': [26],
             'epochs': 800,
             'act_func': ['tanh']
         }

    mods = adv_grid_search_cup(tr_input, tr_target, **params)
    print(mods)


__main__()
