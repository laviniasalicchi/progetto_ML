# -*- coding: utf-8 -*-

# ==============================================================================
# E' una hidden layer
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================

import numpy as np
from neural_net import NeuralNetwork
from trainer import NeuralTrainer


class CrossValidator:

    def __init__(self, neural_trainer):
        self.trainer = neural_trainer


    def k_fold(self, input_vect, target_vect, k=4):
        trainer = self.trainer
        input_size = input_vect.shape[1]
        resto = input_size % k
        fold_size = int(input_size / k)
        start_idx = 0
        acc_list = []
        err_list = []

        print(fold_size)
        print(resto)

        for index in range(1, k + 1):
            if resto != 0:
                end_idx = start_idx + (fold_size + 1) # uso il resto come contatore dei fold che devono avere un elemento in più
                resto = resto - 1
            else:
                end_idx = start_idx + fold_size

            test_kfold = input_vect[:, start_idx:end_idx]
            test_targets = target_vect[:, start_idx:end_idx]

            train_kfold = np.delete(input_vect, np.s_[start_idx:end_idx], axis=1)
            train_targets = np.delete(target_vect, np.s_[start_idx:end_idx], axis=1)

            start_idx = end_idx

            #trainer._train_no_test(train_kfold, train_targets)
            trainer.train_network(train_kfold, train_targets, test_kfold, test_targets)
            test_res = trainer.net.test_network(test_kfold, test_targets)

            err_list.append(test_res[0])
            acc_list.append(test_res[1])

        acc_mean = np.mean(acc_list)
        err_mean = np.mean(err_list)
        # TODO remove printing
        #print(acc_list)
        #print(acc_mean)
        #print(err_list)
        return acc_mean

    @staticmethod
    def k_fold_grid(net_param, train_param, input_vect, target_vect, k=4, **kwargs):
        # parametri topologici rete
        tot_lay = net_param.get('tot_lay', 3)
        units_in = net_param.get('units_in', input_vect.shape[0])
        units_hid = net_param.get('units_hid', 1)
        units_out = net_param.get('units_out', target_vect.shape[0])
        act_func = net_param.get('act_func', 'sigmoid')
        input_size = input_vect.shape[1]
        resto = input_size % k
        fold_size = int(input_size / k)
        start_idx = 0
        acc_list = []
        err_list = []

        #print(fold_size)
        #print(resto)

        for index in range(1, k + 1):
            if resto != 0:
                end_idx = start_idx + (fold_size + 1) # uso il resto come contatore dei fold che devono avere un elemento in più
                resto = resto - 1
            else:
                end_idx = start_idx + fold_size

            net = NeuralNetwork.create_network(tot_lay, units_in, units_hid, units_out, act_func)
            trainer = NeuralTrainer(net, **train_param)

            test_kfold = input_vect[:, start_idx:end_idx]
            test_targets = target_vect[:, start_idx:end_idx]

            train_kfold = np.delete(input_vect, np.s_[start_idx:end_idx], axis=1)
            train_targets = np.delete(target_vect, np.s_[start_idx:end_idx], axis=1)

            start_idx = end_idx

            trainer._train_no_test(train_kfold, train_targets)


            test_res = trainer.net.test_network(test_kfold, test_targets)


            err_list.append(test_res[0])
            acc_list.append(test_res[1])

        acc_mean = np.mean(acc_list)
        err_mean = np.mean(err_list)
        # TODO remove printing
        #print(acc_list)
        #print(acc_mean)
        #print(err_list)
        return acc_mean

    @staticmethod
    def kfold_grid_adv(net_param, train_param, input_vect, target_vect, k=4):
        # parametri topologici rete
        tot_lay = net_param.get('tot_lay', 3)
        act_func = net_param.get('act_func', 'sigmoid')
        init = net_param.get('init', 'def')
        un_lays = net_param.get('un_lays', '')

        input_size = input_vect.shape[1]
        resto = input_size % k
        fold_size = int(input_size / k)
        start_idx = 0
        acc_list = []
        err_list = []
        err_per_epoch_tr = [] #curva di errore tr per ogni fold
        err_per_epoch_vl = [] #curva di errore ts per ogni fold

        #print(fold_size)
        #print(resto)

        for index in range(1, k + 1):
            if resto != 0:
                end_idx = start_idx + (fold_size + 1) # uso il resto come contatore dei fold che devono avere un elemento in più
                resto = resto - 1
            else:
                end_idx = start_idx + fold_size

            net = NeuralNetwork.create_advanced_net(tot_lay, un_lays, act_func, init)
            trainer = NeuralTrainer(net, **train_param)

            test_kfold = input_vect[:, start_idx:end_idx]
            test_targets = target_vect[:, start_idx:end_idx]

            train_kfold = np.delete(input_vect, np.s_[start_idx:end_idx], axis=1)
            train_targets = np.delete(target_vect, np.s_[start_idx:end_idx], axis=1)

            start_idx = end_idx

            # trainer._train_no_test(train_kfold, train_targets) # true to print
            res = trainer.train_network(train_kfold, train_targets,test_kfold, test_targets)
            test_res = trainer.net.test_network(test_kfold, test_targets)




            err_list.append(test_res[0])
            acc_list.append(test_res[1])

        acc_mean = np.mean(acc_list)
        err_mean = np.mean(err_list)

        acc_std = np.std(acc_list)
        err_std = np.std(err_list)

        return err_mean, err_std, err_per_epoch_tr, err_per_epoch_vl

    @staticmethod
    def kfold_grid_adv_plot_info(net_param, train_param, input_vect, target_vect, k=4):
        # parametri topologici rete
        tot_lay = net_param.get('tot_lay', 3)
        act_func = net_param.get('act_func', 'sigmoid')
        init = net_param.get('init', 'def')
        un_lays = net_param.get('un_lays', '')

        input_size = input_vect.shape[1]
        resto = input_size % k
        fold_size = int(input_size / k)
        start_idx = 0
        tr_acc_list = [] # accuracy su tr
        vl_acc_list = [] # accuracy su vl
        tr_err_list = []
        vl_err_list = []
        err_per_epoch_tr = [] # curva di errore tr per ogni fold (lista di lista di errori)
        err_per_epoch_vl = [] # curva di errore ts per ogni fold (lista di lista di errori)


        #print(fold_size)
        #print(resto)

        for index in range(1, k + 1):
            if resto != 0:
                end_idx = start_idx + (fold_size + 1) # uso il resto come contatore dei fold che devono avere un elemento in più
                resto = resto - 1
            else:
                end_idx = start_idx + fold_size

            net = NeuralNetwork.create_advanced_net(tot_lay, un_lays, act_func, init)
            trainer = NeuralTrainer(net, **train_param)
            # dataset splitting
            test_kfold = input_vect[:, start_idx:end_idx]
            test_targets = target_vect[:, start_idx:end_idx]

            train_kfold = np.delete(input_vect, np.s_[start_idx:end_idx], axis=1)
            train_targets = np.delete(target_vect, np.s_[start_idx:end_idx], axis=1)

            start_idx = end_idx

            trainer.train_network(train_kfold, train_targets,test_kfold, test_targets)
            train_res = trainer.get_training_history()

            err_per_epoch_tr.append(train_res['tr_err_h'])
            err_per_epoch_vl.append(train_res['ts_err_h'])
            tr_acc_list.append(train_res['tr_acc'])
            vl_acc_list.append(train_res['ts_acc'])
            tr_err_list.append(train_res['tr_err'])
            vl_err_list.append(train_res['ts_err'])

        # medie
        tr_mean_err = np.mean(tr_err_list)
        vl_mean_err = np.mean(vl_err_list)
        tr_mean_acc = np.mean(tr_acc_list)
        vl_mean_acc = np.mean(vl_acc_list)
        # deviazioni standard
        tr_std_err = np.std(tr_err_list)
        vl_std_err = np.std(vl_err_list)
        tr_std_acc = np.std(tr_acc_list)
        vl_std_acc = np.std(vl_acc_list)
        # risultato
        kfold_res = {
            'tr_mean_err': tr_mean_err,
            'vl_mean_err': vl_mean_err,
            'tr_mean_acc': tr_mean_acc,
            'vl_mean_acc': vl_mean_acc,
            'tr_folds_err_h': err_per_epoch_tr,
            'vl_folds_err_h': err_per_epoch_vl,
            'tr_std_err': tr_std_err,
            'vl_std_err': vl_std_err,
            'tr_std_acc': tr_std_acc,
            'vl_std_acc': vl_std_acc
        }

        return kfold_res
