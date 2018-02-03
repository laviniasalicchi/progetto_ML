# -*- coding: utf-8 -*-

# ==============================================================================
# Oggetto che gestisce il training di una rete
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================

from neural_net import NeuralNetwork
from plotter import Plotter
import random as rn
import numpy as np
"""
La classe Trainer gestisce il training di una rete.
Trainer accetta una lista di parametri passati come dizionario nel costruttore
"""


class NeuralTrainer:

    def __init__(self, neural_net, **kwargs):
        """
        eta: Learning rate
        alfa: Weight decay
        lambd: Tikhonov regularization
        threshold: Soglia dell'errore. Se l'errore scende sotto la soglia
                   si interrompe il training
        loss: funzione di errore della rete Neurale
        """

        self.net = neural_net
        self.eta = kwargs.get('eta', 0.01)
        self.lambd = kwargs.get('lambd', 0.00)
        self.epochs = kwargs.get('epochs', 100)
        self.alfa = kwargs.get('alfa', 0.00)
        self.loss = kwargs.get('loss', "mean_euclidean")
        self.threshold = kwargs.get('treshold', 0.00)
        self.rprop_delt0 = 0.1
        self.rprop_delt_max = 50
        self.tr_err_history = [] # tr error per ogni epochs
        self.ts_err_history = [] # ts error per ogni epochs
        self.tr_acc_history = [] # tr accuracy per ogni epochs
        self.ts_acc_history = [] # ts accuracy per ogni epochs

    def train_network(self, input_vector, target_value, input_test, target_test):

        nn_net = self.net

        if self.loss == 'mean_euclidean':
            loss = NeuralNetwork.mean_euclidean_err
        elif self.loss == 'mean_squared_err':
            loss = NeuralNetwork.mean_squared_err
        else:
            loss = NeuralNetwork.mean_squared_err
            print('WARNING:\t loss function unkown. Defaulted to mean_euclidean')
        nn_net.define_loss(loss)
        tr_errors = [] # training set errors
        tr_accuracy = []
        ts_errors = [] # test set errors
        ts_accuracy = []
        for epoch in range(self.epochs):
            #logger.info("Epoch %s", str(epoch))

            output = nn_net.forward_propagation(input_vector)
            acc = NeuralNetwork.accuracy(output, target_value)

            tr_err = nn_net.backpropagation(input_vector, target_value, loss, self.eta, self.alfa, self.lambd)
            tr_accuracy.append(acc)
            tr_errors.append(tr_err)

            ts_err, ts_acc = nn_net.test_network(input_test, target_test)

            ts_accuracy.append(ts_acc)
            ts_errors.append(ts_err)

        self.tr_err_history = tr_errors
        self.tr_accuracy_history = tr_accuracy
        self.ts_accuracy_history = ts_accuracy
        self.ts_err_history = ts_errors
        return tr_errors

    def _train_no_test(self, input_vector, target_value):

        nn_net = self.net

        if self.loss == 'mean_euclidean':
            loss = NeuralNetwork.mean_euclidean_err
        elif self.loss == 'mean_squared_err':
            loss = NeuralNetwork.mean_squared_err
        else:
            loss = NeuralNetwork.mean_squared_err
            print('WARNING:\t loss function unkown. Defaulted to mean_euclidean')
        nn_net.define_loss(loss)

        tr_errors = [] # training set errors
        tr_accuracy = []

        for epoch in range(self.epochs):
            #logger.info("Epoch %s", str(epoch))

            output = nn_net.forward_propagation(input_vector)
            acc = NeuralNetwork.accuracy(output, target_value)

            tr_err = nn_net.backpropagation(input_vector, target_value, loss, self.eta, self.alfa, self.lambd)
            tr_accuracy.append(acc)
            tr_errors.append(tr_err)

            ts_err, ts_acc = nn_net.test_network(input_test, target_test)
            ts_errors.append(ts_err)

        self.tr_err_history = tr_errors
        self.tr_accuracy_history = tr_accuracy
        return ts_errors

    def train_rprop(self, input_vector, target_value, input_test, target_test):
        nn_net = self.net

        if self.loss == 'mean_euclidean':
            loss = NeuralNetwork.mean_euclidean_err
        elif self.loss == 'mean_squared_err':
            loss = NeuralNetwork.mean_squared_err
        else:
            loss = NeuralNetwork.mean_squared_err
            print('WARNING:\t loss function unkown. Defaulted to mean_euclidean')

        nn_net.define_loss(loss)

        tr_errors = [] # training set errors

        tr_accuracy = []
        ts_errors = []
        ts_accuracy = []
        for epoch in range(self.epochs):
            output = nn_net.forward_propagation(input_vector)
            acc = NeuralNetwork.accuracy(output, target_value)

            err = nn_net.rprop(input_vector, target_value, loss, self.rprop_delt0, self.rprop_delt_max)
            tr_accuracy.append(acc)
            tr_errors.append(err)

            ts_err, ts_acc = nn_net.test_network(input_test, target_test)
            ts_accuracy.append(ts_acc)
            ts_errors.append(ts_err)

        self.tr_err_history = tr_errors
        self.tr_accuracy_history = tr_accuracy
        self.ts_accuracy_history = ts_accuracy
        self.ts_err_history = ts_errors

        print("TR:", err, " - ", acc)
        print("TS:", ts_err, " - ", ts_acc)

    def train_rprop_no_test(self, input_vector, target_value):
        nn_net = self.net
        if self.loss == 'mean_euclidean':
            loss = NeuralNetwork.mean_euclidean_err
        elif self.loss == 'mean_squared_err':
            loss = NeuralNetwork.mean_squared_err
        else:
            loss = NeuralNetwork.mean_squared_err
            print('WARNING:\t loss function unkown. Defaulted to mean_euclidean')
        nn_net.define_loss(loss)

        tr_errors = [] # training set errors
        tr_accuracy = []

        for epoch in range(self.epochs):

            output = nn_net.forward_propagation(input_vector)
            acc = NeuralNetwork.accuracy(output, target_value)

            err = nn_net.rprop(input_vector, target_value, loss, self.rprop_delt0, self.rprop_delt_max)
            tr_accuracy.append(acc)
            tr_errors.append(err)

        self.tr_err_history = tr_errors
        self.tr_accuracy_history = tr_accuracy


    def get_training_history(self):
        final_ts_err = 0
        final_ts_acc = 0
        final_tr_err = 0
        final_tr_acc = 0

        if len(self.ts_err_history) > 0:
            final_ts_err = self.ts_err_history[len(self.ts_err_history) - 1]
        if len(self.ts_acc_history) > 0:
            final_ts_acc = self.ts_acc_history[len(self.ts_acc_history) - 1]
        if len(self.tr_err_history) > 0:
            final_tr_err = self.tr_err_history[len(self.tr_err_history) - 1]
        if len(self.tr_acc_history) > 0:
            final_tr_acc = self.tr_acc_history[len(self.tr_acc_history) - 1]

        training_hist = {
            'tr_err_h': self.tr_err_history,
            'ts_err_h': self.ts_err_history,
            'tr_acc_h': self.tr_acc_history,
            'ts_acc_h': self.ts_acc_history,
            'tr_acc': final_tr_acc,
            'ts_acc': final_ts_acc,
            'tr_err': final_tr_err,
            'ts_err': final_ts_err
        }
        return training_hist


    # @staticmethod
    # def shuffles(input_d, target_d):
    #     copy_in = input_d.copy()
    #     copy_tar = target_d.copy()
    #     new_input = np.zeros((input_d.shape[0], input_d.shape[1]))
    #     new_target = np.zeros((target_d.shape[0], target_d.shape[1]))
    #     col = 0
    #     while copy_in!=[]:
    #         indx = rn.randint(0, copy_in.shape[1]-1)
    #         new_input[:,col]=copy_in[:,indx]
    #         copy_in = np.delete(copy_in, indx, axis=1)
    #         new_target[:,col] = copy_tar[:, indx]
    #         copy_tar = np.delete(copy_tar, indx, axis=1)
    #         col=col+1
    #     new_input[:, col] = copy_in[:,0]
    #     new_target[:, col] = copy_tar[:,0]
    #     return new_input, new_target
