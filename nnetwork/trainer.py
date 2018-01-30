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

    def train_network(self, input_vector, target_value, input_test, target_test, folder, save=False):

        nn_net = self.net

        if self.loss == 'mean_euclidean':
            loss = NeuralNetwork.mean_euclidean_err
        elif self.loss == 'mean_squared_err':
            loss = NeuralNetwork.mean_squared_err
        else:
            loss = NeuralNetwork.mean_squared_err
            print('WARNING:\t loss function unkown. Defaulted to mean_euclidean')
        errors = []
        accuracy = []
        epochs_plot = []
        ts_errors = []
        ts_accuracy = []
        weights_BT = {}  # // dizionario inizialmente vuoto per salvare il modello con l'errore più basso
        err_BT = 4.51536876901e+19  # // errore con valore inizialmente enorme, servirà per il backtracking
        for epoch in range(self.epochs):
            #logger.info("Epoch %s", str(epoch))

            output = nn_net.forward_propagation(input_vector)
            acc = NeuralNetwork.accuracy(output, target_value)

            err = nn_net.backpropagation(input_vector, target_value, loss, self.eta, self.alfa, self.lambd)
            accuracy.append(acc)
            errors.append(err)

            ts_err, ts_acc = nn_net.test_network(input_test, target_test)
            ts_accuracy.append(ts_acc)
            ts_errors.append(ts_err)

            epochs_plot.append(epoch)

            # // creazione dizionario {nomelayer : pesi}
            for i in range(len(nn_net.hidden_layers)):
                layer = nn_net.hidden_layers[i]
                if i == 0:
                    # // weights è un dizionario per poter avere i pesi aggiornati raggiungibili dal nome del layer
                    key = "hidden" + str(i)
                    weights = ({key: layer.weights})
                else:
                    key = "hidden" + str(i)
                    weights.update({key: layer.weights})
            weights.update({'output': nn_net.output_layer.weights})

            # // se l'errore scende sotto la soglia, si salva il modello che lo produce
            if err < self.threshold:
                print('lavinia puzzecchia! trallallero taralli e vino')
                #   NeuralNetwork.saveModel(self, weights)
                break
            # // se l'errore del modello di turno supera il precedente, si sovrascrive a weights il modello precedente
            # WARNING: l'errore può avere minimi locali, più avanti definiremo meglio questo if
            elif err > err_BT:
                weights = weights_BT
                #   NeuralNetwork.saveModel(self, weights)
            # // altrimenti, se l'errore continua a decrescere, si sovrascrive a weights_BT il modello corrente, si salva e si sovrascrive a error_BT l'errore del modello corrente
            else:
                weights_BT = weights
                err_BT = err
        print("TR",err, " - ", acc)
        print("TS", ts_err, " - ", ts_acc)
        if save:
            # todo parte di salvetaggio del modello

            Plotter.plotError(epochs_plot, errors, ts_errors, folder)
            Plotter.plot_accuracy(epochs_plot, accuracy, ts_accuracy, folder)
        return weights, err

    def _train_no_test(self, input_vector, target_value, save=False):
        nn_net = self.net

        if self.loss == 'mean_euclidean':
            loss = NeuralNetwork.mean_euclidean_err
        elif self.loss == 'mean_squared_err':
            loss = NeuralNetwork.mean_squared_err
        else:
            loss = NeuralNetwork.mean_squared_err
            print('WARNING:\t loss function unkown. Defaulted to mean_euclidean')
        errors = []
        accuracy = []
        epochs_plot = []
        ts_errors = []
        ts_accuracy = []
        weights_BT = {}  # // dizionario inizialmente vuoto per salvare il modello con l'errore più basso
        err_BT = 4.51536876901e+19  # // errore con valore inizialmente enorme, servirà per il backtracking

        for epoch in range(self.epochs):

            mb = 8
            batch_size = int(input_vector.shape[1] / mb)
            start_idx = 0
            for minib in range(1, mb + 1):
                end_idx = start_idx + batch_size
                input_mb = input_vector[:, start_idx:end_idx]
                target_mb = target_value[:, start_idx:end_idx]
                input_mb, target_mb = NeuralTrainer.shuffles(input_mb, target_mb)
                output = nn_net.forward_propagation(input_mb)
                acc = NeuralNetwork.accuracy(output, target_mb)
                err = nn_net.backpropagation(input_mb, target_mb, loss, self.eta, self.alfa, self.lambd)
                start_idx = end_idx

            #   output = nn_net.forward_propagation(input_vector)
            #   acc = NeuralNetwork.accuracy(output, target_value)
            #   err = nn_net.backpropagation(input_vector, target_value, loss, self.eta, self.alfa, self.lambd)
            accuracy.append(acc)
            errors.append(err)
            epochs_plot.append(epoch)

            # // creazione dizionario {nomelayer : pesi}
            for i in range(len(nn_net.hidden_layers)):
                layer = nn_net.hidden_layers[i]
                if i == 0:
                    # // weights è un dizionario per poter avere i pesi aggiornati raggiungibili dal nome del layer
                    key = "hidden" + str(i)
                    weights = ({key: layer.weights})
                else:
                    key = "hidden" + str(i)
                    weights.update({key: layer.weights})
            weights.update({'output': nn_net.output_layer.weights})

            # // se l'errore scende sotto la soglia, si salva il modello che lo produce
            if err < self.threshold:
                print('lavinia puzzecchia! trallallero taralli e vino')
                #   NeuralNetwork.saveModel(self, weights)
                break
            # // se l'errore del modello di turno supera il precedente, si sovrascrive a weights il modello precedente
            # WARNING: l'errore può avere minimi locali, più avanti definiremo meglio questo if
            elif err > err_BT:
                weights = weights_BT
                #   NeuralNetwork.saveModel(self, weights)
            # // altrimenti, se l'errore continua a decrescere, si sovrascrive a weights_BT il modello corrente, si salva e si sovrascrive a error_BT l'errore del modello corrente
            else:
                weights_BT = weights
                err_BT = err
        if save:
            Plotter.plotError_noTS(epochs_plot, errors)
            #Plotter.plot_accuracy_noTS(epochs_plot, accuracy)
        #print("Accuracy;", accuracy[len(accuracy) - 1])
        print("err\n", err)
        return weights, err

    def train_rprop(self, input_vector, target_value, input_test, target_test):
        nn_net = self.net

        if self.loss == 'mean_euclidean':
            loss = NeuralNetwork.mean_euclidean_err
        elif self.loss == 'mean_squared_err':
            loss = NeuralNetwork.mean_squared_err
        else:
            loss = NeuralNetwork.mean_squared_err
            print('WARNING:\t loss function unkown. Defaulted to mean_euclidean')
        errors = []
        accuracy = []
        epochs_plot = []
        ts_errors = []
        ts_accuracy = []
        err_BT = 4.51536876901e+19  # // errore con valore inizialmente enorme, servirà per il backtracking
        for epoch in range(self.epochs):
            output = nn_net.forward_propagation(input_vector)
            acc = NeuralNetwork.accuracy(output, target_value)

            err = nn_net.rprop(input_vector, target_value, loss, self.rprop_delt0, self.rprop_delt_max)
            accuracy.append(acc)
            errors.append(err)

            ts_err, ts_acc = nn_net.test_network(input_test, target_test)
            ts_accuracy.append(ts_acc)
            ts_errors.append(ts_err)

            epochs_plot.append(epoch)
        print("TR:", err, " - ", acc)
        print("TS:", ts_err, " - ", ts_acc)
        Plotter.plotError(epochs_plot, errors, ts_errors, "d/")
        Plotter.plot_accuracy(epochs_plot, accuracy, ts_accuracy, "d/")

    def train_rprop_no_test(self, input_vector, target_value):
        nn_net = self.net
        if self.loss == 'mean_euclidean':
            loss = NeuralNetwork.mean_euclidean_err
        elif self.loss == 'mean_squared_err':
            loss = NeuralNetwork.mean_squared_err
        else:
            loss = NeuralNetwork.mean_squared_err
            print('WARNING:\t loss function unkown. Defaulted to mean_euclidean')
        errors = []
        accuracy = []
        epochs_plot = []

        err_BT = 4.51536876901e+19  # // errore con valore inizialmente enorme, servirà per il backtracking
        for epoch in range(self.epochs):
            '''mb = 8
            batch_size = int(input_vector.shape[1] / mb)
            start_idx = 0
            for minib in range(1, mb + 1):
                end_idx = start_idx + batch_size
                input_mb = input_vector[:, start_idx:end_idx]
                target_mb = target_value[:, start_idx:end_idx]
                output = nn_net.forward_propagation(input_mb)
                acc = NeuralNetwork.accuracy(output, target_mb)
                err = nn_net.rprop(input_mb, target_mb, loss, self.rprop_delt0, self.rprop_delt_max)
                start_idx = end_idx'''
            output = nn_net.forward_propagation(input_vector)
            acc = NeuralNetwork.accuracy(output, target_value)

            err = nn_net.rprop(input_vector, target_value, loss, self.rprop_delt0, self.rprop_delt_max)
            accuracy.append(acc)
            errors.append(err)
            epochs_plot.append(epoch)
        Plotter.plotError_noTS(epochs_plot, errors)
        Plotter.plot_accuracy_noTS(epochs_plot, accuracy)

    @staticmethod
    def shuffles(input_d, target_d):
        copy_in = input_d.copy()
        copy_tar = target_d.copy()
        new_input = np.zeros((input_d.shape[0], input_d.shape[1]))
        new_target = np.zeros((target_d.shape[0], target_d.shape[1]))
        col = 0
        while copy_in!=[]:
            indx = rn.randint(0, copy_in.shape[1]-1)
            new_input[:,col]=copy_in[:,indx]
            copy_in = np.delete(copy_in, indx, axis=1)
            new_target[:,col] = copy_tar[:, indx]
            copy_tar = np.delete(copy_tar, indx, axis=1)
            col=col+1
        new_input[:, col] = copy_in[:,0]
        new_target[:, col] = copy_tar[:,0]
        return new_input, new_target