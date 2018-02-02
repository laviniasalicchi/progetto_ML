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
            res = trainer.train_network(train_kfold, train_targets,test_kfold, test_targets, 'none')
            test_res = trainer.net.test_network(test_kfold, test_targets)




            err_list.append(test_res[0])
            acc_list.append(test_res[1])

        acc_mean = np.mean(acc_list)
        err_mean = np.mean(err_list)

        acc_std = np.std(acc_list)
        err_std = np.std(err_list)

        err_per_epoch_tr.append(res[3])
        err_per_epoch_vl.append(res[2])
        #print(err_per_epoch_vl)



        return err_mean, err_std, err_per_epoch_tr, err_per_epoch_vl
