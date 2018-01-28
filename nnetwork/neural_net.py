# -*- coding: utf-8 -*-

# ==============================================================================
# E' una Rete Neurale di reti neurali neurali anch'esse
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================

from input_layer import InputLayer
# from save-load modello import Save, Load
import numpy as np
import os
from datetime import datetime
from layer import Layer
from input_layer import InputLayer
from hidden_layer import HiddenLayer
from output_layer import OutputLayer

import matplotlib
matplotlib.use('TkAgg')  # mac osx need this backend

import matplotlib.pyplot as plt
import re
import logging
import sys


class NeuralNetwork:

    def __init__(self):
        self.input_layer = []
        self.hidden_layers = []
        self.output_layer = []
        self.logger = logging.getLogger(__name__)

    def add_input_layer(self, input_layer):
        """
        Aggiunge un input layer alla rete.

        Se il layer precedente viene sovrascritto stampa un warning
        """
        if isinstance(input_layer, list):
            self.logger.debug('Input layer connected')
        else:
            self.logger.debug('Previous input layer overwritten')
        self.input_layer = input_layer

    def add_hidden_layer(self, hidden_layer):
        """Aggiunge un hidden layer alla lista di hidden layers."""
        self.hidden_layers.append(hidden_layer)

    def remove_hidden_layer(self, index):
        """Rimuove l'hidden layer alla posizione index della lista"""
        self.hidden_layers.pop(index)

    def add_output_layer(self, output_layer):
        """Aggiunge un layer di output alla rete"""
        self.output_layer = output_layer

    def define_loss(self, loss_function):
        """Specificare quale loss function utilizzare"""
        self.loss_function = loss_function

    @staticmethod
    def create_network(
            tot_layers,
            units_in,
            units_hid,
            units_out,
            activ_func,
            slope=1
            ):
        """
        Crea una rete neurale in base ai parametri.

            tot_layers = numero di layer totali della rete
            units_in = numero di unità dell'input layers
            units_hid = numero di unità di ogni hidden layer
            units_out = numero di unità dell'output layer
            activ_func = nome della funzione di attivazione da utilizzare
            slope = pendenza della funzione di attivazione. Usato per sigmoid
        """

        neural_network = NeuralNetwork()
        hidden_num = tot_layers - 2
        input_layer = InputLayer(units_in)
        input_layer.create_weights(units_in)
        input_layer.set_activation_function(activ_func)
        input_layer.set_sigmoid_slope(slope)
        neural_network.add_input_layer(input_layer)

        hidden_l = HiddenLayer(units_hid)
        hidden_l.create_weights(units_in)
        hidden_l.set_activation_function(activ_func)
        neural_network.add_hidden_layer(hidden_l)

        for i in range(1, hidden_num):
            hidden_l = HiddenLayer(units_hid)
            hidden_l.create_weights(units_hid)
            hidden_l.set_activation_function(activ_func)
            hidden_l.set_sigmoid_slope(slope)
            neural_network.add_hidden_layer(hidden_l)

        output_layer = OutputLayer(units_out)
        output_layer.create_weights(units_hid)
        output_layer.set_activation_function(activ_func)
        output_layer.set_sigmoid_slope(slope)
        neural_network.add_output_layer(output_layer)
        #neural_network.print_net_info()
        return neural_network

    @staticmethod
    def create_advanced_net(tot_lays, un_lay, afs, init):
        """
        Crea una rete neurale con una topologia più complessa.
            tot_lays = numero di layer totali
            un_lay = lista del numero di unità per layer. Il primo elemento
                     è associato al primo layer. Il numero di elementi della
                     lista deve essere uguale al numero di layers
            afs = lista di funzioni di attivazione: una per ogni layer.
                  Il numero di elementi della
                  lista deve essere uguale al numero di layers
            init = tipo di inizializzazione dei pesi.
                    xavier = fan_in
                    Understanding the difficulty of training deep
                    feedforward neural networks
                    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        """

        logger = logging.getLogger(__name__)
        if tot_lays != len(un_lay):
            warn = ('Il numero di layer e la lista di unità per layer'
                    'non coincidono. Specificare un numero di unità '
                    'per ogni layer della rete')
            logger.error(warn)
            return
        elif tot_lays != len(afs):
            warn = ('Numero di layer e lista di funzioni di attivazione '
                    'non hanno la stessa lunghezza.')
            logger.error(warn)
        if tot_lays == 2:
            logger.error('Il minimo numero di layer è 3')
            return

        slope = 1
        net = NeuralNetwork()
        fan_in = 2 / (un_lay[0] + un_lay[len(un_lay) - 1])
        hidden_num = tot_lays - 2
        input_layer = InputLayer(un_lay[0])

        if init == 'xavier':
            input_layer.create_weights_fan_in(un_lay[0], fan_in)
        else:
            input_layer.create_weights(un_lay[0])

        input_layer.set_activation_function(afs[0])
        net.add_input_layer(input_layer)

        prev_un = un_lay[0]

        for i in range(1, hidden_num + 1):
            hidden_l = HiddenLayer(un_lay[i])
            if init == 'xavier':
                hidden_l.create_weights_fan_in(prev_un, fan_in)
            else:
                hidden_l.create_weights(prev_un)
            prev_un = un_lay[i]

            hidden_l.set_activation_function(afs[i])
            hidden_l.set_sigmoid_slope(slope)
            net.add_hidden_layer(hidden_l)

        last_idx = len(un_lay) - 1
        output_layer = OutputLayer(un_lay[last_idx])
        output_layer.create_weights_fan_in(un_lay[last_idx - 1], fan_in)
        output_layer.set_activation_function(afs[last_idx])
        output_layer.set_sigmoid_slope(slope)
        net.add_output_layer(output_layer)
        #net.print_net_info()
        return net


    def predict(self, input_vector):
        return forward_propagation(self, input_vector)

    def print_net_info(self):
        """Stampa informazioni sulla topologia della rete"""

        print('**************** NETWORK ****************')
        print('Input Layer: ' + str(self.input_layer.n_units) + ' units')
        i = 1
        for h_layer in self.hidden_layers:
            print('HiddenLayer ' + str(i) + ': ' + str(h_layer.n_units) + ' units')
            i = i + 1
        print('Output Layer: ' + str(self.output_layer.n_units) + ' units')
        print('**************** NETWORK ***************')

    def forward_propagation(self, input_vector):
        """
        Implementa la forward propagation.

        Il segnale di input viene propagato in avanti e viene calcolato
        l'output di ogni layer.
        """
        net = self.input_layer.net_function(input_vector)
        input_layer_out = self.input_layer.layer_output()
        self.input_layer.output = input_layer_out

        if len(self.hidden_layers) <= 1:

            h_layer = self.hidden_layers[0]
            h_layer.net = h_layer.net_function(input_layer_out)
            h_layer_out = h_layer.layer_output()
            h_layer.output = h_layer_out
            self.output_layer.net = self.output_layer.net_function(h_layer_out)
            out_layer_out = self.output_layer.layer_output()
            self.output_layer.output = out_layer_out

        else:
            last_layer_out = input_layer_out
            for h_layer in self.hidden_layers:
                h_layer.net = h_layer.net_function(last_layer_out)
                last_layer_out = h_layer.layer_output()
                h_layer.output = last_layer_out

            self.output_layer.net = self.output_layer.net_function(last_layer_out)
            out_layer_out = self.output_layer.layer_output()
            self.output_layer.output = out_layer_out

        return self.output_layer.output


    def backpropagation(self, input_vector, target_value, err_func, eta, alfa, lambd):
        """
        eta = learning rate
        alfa = momentum
        lambda = Tikhonov regularization
        """
        # delt = deriv(E/out) * f'(net)
        err_deriv = err_func(target_value, self.output_layer.output, True)
        self.logger.debug('Backprop: err_deriv.shape %s', str(err_deriv.shape))

        out_net = self.output_layer.net

        self.logger.debug("Backprop: out_net.shape %s", str(out_net.shape))

        f_prime = self.output_layer.activation_function_derivative(out_net)
        self.logger.debug("Backprop: f_prime shape %s", str(f_prime.shape))

        delta_out = err_deriv * f_prime  # dovrebbe essere una matrice con colonne = numero di pattern // è pattern x n output units
        self.output_layer.deltas = delta_out
        prev_layer_delta = delta_out
        prev_layer_weights = self.output_layer.weights  # prev layer weights sono i pesi del layer precedente (quindi quello a destra quando si fa la backprop)
        self.logger.debug("Backprop: delta_out.shape %s", str(delta_out.shape))

        for layer in reversed(self.hidden_layers):
            layer_net = layer.net
            self.logger.debug("Backprop: layer_net.shape %s", str(layer_net.shape))

            f_prime = layer.activation_function_derivative(layer_net)

            prev_layer_weights = np.delete(prev_layer_weights, -1, 0)  # tolta la riga dei pesi del bias
            transpose_weights = np.transpose(prev_layer_weights)  # // trasposta fatta a parte senza motivo

            self.logger.debug("Backprop: prev_layer_weights.shape %s", str(prev_layer_weights.shape))
            self.logger.debug("Backprop: prev_layer_delta.shape %s", str(prev_layer_weights.shape))
            self.logger.debug("Backprop: f_prime.shape %s", str(f_prime.shape))

            delta = np.dot(prev_layer_weights, prev_layer_delta) * f_prime

            layer.deltas = delta

            self.logger.debug("Backprop: layer_deltas.shape %s", str(layer.deltas.shape))

            prev_layer_delta = delta
            prev_layer_weights = layer.weights

        # update weights
        self.logger.debug("Backprop: layer.weights.shape %s", str(layer.weights.shape))
        ones_row = np.ones((1, layer.deltas.shape[1]))

        # update weights
        # d(E)/d(w_ji) = sum_p(delta_j * out_i)
        last_layer_out = self.input_layer.output
        net_layers = []
        for h_layer in self.hidden_layers:
            net_layers.append(h_layer)
        net_layers.append(self.output_layer)


        for layer in net_layers:
            dW = np.dot(last_layer_out, layer.deltas.T)

            momentum = layer.last_dW * alfa
            reg_term = (lambd * layer.weights)

            layer.weights = layer.weights - (eta * dW) + momentum - reg_term  # +/- 2*lambda*layer.weights (per Tikhonov reg.)  //  + (alfa * prev_layer_delta)  (per momentum)
            # print("DW pre", layer.last_dW)
            layer.last_dW = - (eta * dW) + momentum
            # print("DW post", layer.last_dW)
            last_layer_out = layer.output


        error = err_func(target_value, self.output_layer.output)


        return error


    def rprop(self, input_vect, target_value, err_func, delt0, delt_max):

        npos = 1.2
        nneg = 0.5
        delt_max = 50.0
        delt_min = 1.0e-6

        logger = logging.getLogger(__name__)

        # delt = deriv(E/out) * f'(net)
        err_deriv = err_func(target_value, self.output_layer.output, True)
        logger.debug('Rprop: err_deriv.shape %s', str(err_deriv.shape))

        out_net = self.output_layer.net

        logger.debug("Rprop: out_net.shape %s", str(out_net.shape))

        f_prime = self.output_layer.activation_function_derivative(out_net)
        logger.debug("Rprop: f_prime shape %s", str(f_prime.shape))

        delta_out = err_deriv * f_prime  # dovrebbe essere una matrice con colonne = numero di pattern // è pattern x n output units
        self.output_layer.deltas = delta_out
        prev_layer_delta = delta_out
        prev_layer_weights = self.output_layer.weights  # prev layer weights sono i pesi del layer precedente (quindi quello a destra quando si fa la Bprop)
        logger.debug("Rprop: delta_out.shape %s", str(delta_out.shape))

        for layer in reversed(self.hidden_layers):
            layer_net = layer.net
            logger.debug("Rprop: layer_net.shape %s", str(layer_net.shape))

            f_prime = layer.activation_function_derivative(layer_net)

            prev_layer_weights = np.delete(prev_layer_weights, -1, 0)  # tolta la riga dei pesi del bias
            transpose_weights = np.transpose(prev_layer_weights)  # // trasposta fatta a parte senza motivo

            logger.debug("Rprop: prev_layer_weights.shape %s", str(prev_layer_weights.shape))
            logger.debug("Rprop: prev_layer_delta.shape %s", str(prev_layer_weights.shape))
            logger.debug("Rprop: f_prime.shape %s", str(f_prime.shape))

            delta = np.dot(prev_layer_weights, prev_layer_delta) * f_prime

            layer.deltas = delta

            logger.debug("Rprop: layer_deltas.shape %s", str(layer.deltas.shape))

            prev_layer_delta = delta
            prev_layer_weights = layer.weights

        last_layer_out = self.input_layer.output
        net_layers = []
        for h_layer in self.hidden_layers:
            net_layers.append(h_layer)
        net_layers.append(self.output_layer)

        for layer in net_layers:
            # dW = δE/δwji
            dW = np.dot(last_layer_out, layer.deltas.T)
            #print(dW.shape)

            last_dW = np.zeros(dW.shape) # matrice dei δE/δwji a t-1
            sum_W = np.zeros(dW.shape)  # matrice dei pesi da sommare alla vechhia

            #  moltiplico δE/δwji(t) e δE/δwji (t-1) elementWise
            err_prod = np.multiply(dW, layer.last_dW)
            #print(err_prod.shape)
            #print(range(0, len(err_prod[0])))
            for i in range(0, err_prod.shape[0]):  # righe
                for j in range(0, err_prod.shape[1]):  # colonne
                    #print("i",i,"j",j)
                    if err_prod[i][j] > 0:
                        delt_ij = min(layer.delta_rprop[i][j] * npos, delt_max)
                        layer.delta_rprop[i][j] = delt_ij
                        delta_wij = -1 * np.sign(dW[i][j]) * delt_ij
                        sum_W[i][j] = delta_wij
                        last_dW[i][j] = dW[i][j]
                    elif err_prod[i][j] < 0:
                        delt_ij = max(layer.delta_rprop[i][j] * nneg, delt_min)
                        layer.delta_rprop[i][j] = delt_ij
                        last_dW[i][j] = 0
                        sum_W[i][j] = 0
                    elif err_prod[i][j] == 0:
                        delta_wij = -1 * np.sign(dW[i][j]) * layer.delta_rprop[i][j]
                        sum_W[i][j] = delta_wij
                        last_dW[i][j] = dW[i][j]
            layer.last_dW = last_dW
            layer.weights = layer.weights + sum_W
            last_layer_out = layer.output

        return err_func(target_value, self.output_layer.output)

    def train_rprop(self, input_vector, target_value, input_test, target_test, epochs, threshold, loss_func, delt0, delt_max):
        logger = logging.getLogger(__name__)
        loss = NeuralNetwork.mean_euclidean_err
        if loss_func == 'mean_euclidean':
            loss = NeuralNetwork.mean_euclidean_err
        elif loss_func == 'mean_squared_err':
            loss = NeuralNetwork.mean_squared_err
        else:
            print('WARNING:\t loss function unkown. Defaulted to mean_euclidean')
        errors = []
        accuracy = []
        epochs_plot = []
        ts_errors = []
        ts_accuracy = []
        weights_BT = {}  # // dizionario inizialmente vuoto per salvare il modello con l'errore più basso
        err_BT = 4.51536876901e+19  # // errore con valore inizialmente enorme, servirà per il backtracking
        for epoch in range(epochs):
            logger.info("Epoch %s", str(epoch))
            forward_prop = NeuralNetwork.forward_propagation(self, input_vector)
            acc = NeuralNetwork.accuracy(self.output_layer.output, target_value)

            err = NeuralNetwork.rprop(self, input_vector, target_value, loss, delt0, delt_max)
            accuracy.append(acc)
            errors.append(err)

            ts_err, ts_acc = NeuralNetwork.test_network(self, input_test, target_test)
            ts_accuracy.append(ts_acc)
            ts_errors.append(ts_err)

            epochs_plot.append(epoch)

        NeuralNetwork.plotError(self, epochs_plot, errors, ts_errors)
        NeuralNetwork.plot_accuracy(self, epochs_plot, accuracy, ts_accuracy)

    def _train_no_test(self, input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd, final=False):
        logger = logging.getLogger(__name__)
        loss = NeuralNetwork.mean_euclidean_err
        if loss_func == 'mean_euclidean':
            loss = NeuralNetwork.mean_euclidean_err
        elif loss_func == 'mean_squared_err':
            loss = NeuralNetwork.mean_squared_err
        else:
            print('WARNING:\t loss function unkown. Defaulted to mean_euclidean')
        errors = []
        accuracy = []
        epochs_plot = []
        ts_errors = []
        ts_accuracy = []
        weights_BT = {}  # // dizionario inizialmente vuoto per salvare il modello con l'errore più basso
        err_BT = 4.51536876901e+19  # // errore con valore inizialmente enorme, servirà per il backtracking
        for epoch in range(epochs):
            logger.info("Epoch %s", str(epoch))
            forward_prop = NeuralNetwork.forward_propagation(self, input_vector)
            acc = NeuralNetwork.accuracy(self.output_layer.output, target_value)

            err = NeuralNetwork.backpropagation(self, input_vector, target_value, loss, eta, alfa, lambd)
            accuracy.append(acc)
            errors.append(err)
            epochs_plot.append(epoch)

            # // creazione dizionario {nomelayer : pesi}
            for i in range(len(self.hidden_layers)):
                layer = self.hidden_layers[i]
                if i == 0:
                    # // weights è un dizionario per poter avere i pesi aggiornati raggiungibili dal nome del layer
                    key = "hidden" + str(i)
                    weights = ({key: layer.weights})
                else:
                    key = "hidden" + str(i)
                    weights.update({key: layer.weights})
            weights.update({'output': self.output_layer.weights})

            # // se l'errore scende sotto la soglia, si salva il modello che lo produce
            if err < threshold:
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

        if final:
            NeuralNetwork.plotError(self, epochs_plot, errors, ts_errors)
            NeuralNetwork.plot_accuracy(self, epochs_plot, accuracy, ts_accuracy)
        print("Accuracy;", accuracy[len(accuracy) - 1])
        return weights, err





    def train_network(self, input_vector, target_value, input_test, target_test, epochs, threshold, loss_func, eta, alfa, lambd,
                      final=False):  # // aggiunti i target_values
        logger = logging.getLogger(__name__)
        loss = NeuralNetwork.mean_euclidean_err
        if loss_func == 'mean_euclidean':
            loss = NeuralNetwork.mean_euclidean_err
        elif loss_func == 'mean_squared_err':
            loss = NeuralNetwork.mean_squared_err
        else:
            print('WARNING:\t loss function unkown. Defaulted to mean_euclidean')
        errors = []
        accuracy = []
        epochs_plot = []
        ts_errors = []
        ts_accuracy = []
        weights_BT = {}  # // dizionario inizialmente vuoto per salvare il modello con l'errore più basso
        err_BT = 4.51536876901e+19  # // errore con valore inizialmente enorme, servirà per il backtracking
        for epoch in range(epochs):
            logger.info("Epoch %s", str(epoch))
            forward_prop = NeuralNetwork.forward_propagation(self, input_vector)
            acc = NeuralNetwork.accuracy(self.output_layer.output, target_value)

            err = NeuralNetwork.backpropagation(self, input_vector, target_value, loss, eta, alfa, lambd)
            accuracy.append(acc)
            errors.append(err)

            ts_err, ts_acc = NeuralNetwork.test_network(self, input_test, target_test)
            ts_accuracy.append(ts_acc)
            ts_errors.append(ts_err)

            epochs_plot.append(epoch)

            """sys.stdout.write('\r')
            j = (epoch + 1 / epochs)
            sys.stdout.write("[%-20s] %d%%" % ('='*int(j), 100*j))
            sys.stdout.flush()"""


            # // creazione dizionario {nomelayer : pesi}
            for i in range(len(self.hidden_layers)):
                layer = self.hidden_layers[i]
                if i == 0:
                    # // weights è un dizionario per poter avere i pesi aggiornati raggiungibili dal nome del layer
                    key = "hidden" + str(i)
                    weights = ({key: layer.weights})
                else:
                    key = "hidden" + str(i)
                    weights.update({key: layer.weights})
            weights.update({'output': self.output_layer.weights})

            # // se l'errore scende sotto la soglia, si salva il modello che lo produce
            if err < threshold:
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



            '''
            NB: l'errore nel training non dovrebbe mai aumentare col passare delle epoch
            la precedente serie di if è però riutilizzabile quando guardiamo l'errore sul test set
            '''

        # NeuralNetwork.saveModel(self, weights)
        # NeuralNetwork.saveModel(self, weights)
        # // in ogni caso si plotta l'andamento dell'errore su tutte le epoch
        if final:
            NeuralNetwork.plotError(self, epochs_plot, errors, ts_errors)
            NeuralNetwork.plot_accuracy(self, epochs_plot, accuracy, ts_accuracy)
        print("Accuracy;", accuracy[len(accuracy) - 1])
        return weights, err

    """
    tests the network
    return values: accuracy = network accuracy
                   error    = network error
    """
    def test_network(self, x, target_value):
        # solo forward + calcolo dell'errore
        NeuralNetwork.forward_propagation(self, x)
        error = NeuralNetwork.mean_euclidean_err(target_value, self.output_layer.output)
        accuracy = NeuralNetwork.accuracy(self.output_layer.output, target_value)
        return error, accuracy

    @staticmethod
    def accuracy(output_net, target, tanh=True):
        if tanh:
            out_rounded = np.rint(output_net)
            print(output_net)
            print("---------------------------------")
            print(out_rounded)
            out_r = out_rounded.copy()
            out_r = (out_r > 0).astype(int)
            result = np.where(out_r == target, 1, 0)
            result = np.mean(result)
        else:
            out_rounded = np.rint(output_net)
            result = np.where(out_rounded == target, 1, 0)
            result = np.mean(result)
        return result

    @staticmethod
    def test_existing_model(input_test, target_test):
        path = "models/finals/"
        dirs = os.listdir(path)
        for dir in dirs:
            print("********", dir)
            path_hyper = path + dir + "/hyperpar.npz"
            npzfile = np.load(path_hyper)
            eta = npzfile['eta']
            alfa = npzfile['alfa']
            labd = npzfile['lambd']
            ntl = npzfile['ntl']
            nhu = npzfile['nhu']
            af = npzfile['af']
            neural_net = NeuralNetwork.create_network(ntl, 17, nhu, 1, af, slope=1)
            print(ntl, "hiddens", neural_net.hidden_layers)

            dir_wei = path + dir + "/weights"
            wei_files = os.listdir(dir_wei)
            i = 0
            for file in wei_files:
                print(dir_wei, "FILES", file)
                match_hidden = re.match(r'hidden([0-9]).npz', file)
                if match_hidden:
                    print("hidden ok")
                    print(file)
                    fileout = dir_wei + "/" + file
                    npzfile = np.load(fileout)
                    hidden_wei = npzfile['weights']
                    neural_net.hidden_layers[i].weights = hidden_wei
                    i = i + 1
                if file == 'output.npz':
                    print("output ok")
                    fileout = dir_wei + "/" + file
                    npzfile = np.load(fileout)
                    output_wei = npzfile['weights']
                    neural_net.output_layer.weights = output_wei

            neural_net.forward_propagation(input_test)
            ts_err, ts_acc = neural_net.test_network(input_test, target_test)
            print("Error su test set", ts_err)
            print("Accuracy su test set", ts_acc)

            print("************ fine di ", dir)


    """
    MSE - sicuramente sbagliato
        per regolarizzazione: aggiungere +lambda*(weights)**2
    """
    @staticmethod
    def mean_squared_err(target_value, neuron_out, deriv=False):
        if deriv:
            return - (np.subtract(target_value, neuron_out))
        res = np.subtract(target_value, neuron_out) ** 2
        res = np.sum(res, axis=1)
        return res / target_value.shape[1]


    """
    Calcola il MEE
    target_value = matrice che ha per righe i target e come colonne i pattern
    neurons_out = matrice che ha per righe gli output e come colonne i pattern
    divide il risultato per il numero di colonne di target value che dovrebbe
    """
    @staticmethod
    def mean_euclidean_err(target_value, neurons_out, deriv=False):
        if deriv:
            err = NeuralNetwork.mean_euclidean_err(target_value, neurons_out)
            return np.subtract(neurons_out, target_value) * (1 / err)
        res = np.subtract(neurons_out, target_value) ** 2  # matrice con righe = numero neuroni e colonne = numero di pattern  // è al contrario
        res = np.sqrt(res)
        res = np.sum(res,                     axis=0)  # somma sulle colonne. ora res = vettore con 1 riga e colonne = numero di pattern. ogni elemento è (t-o)^2
        res = np.sum(res, axis=0)  # somma sulle righe
        return (res / target_value.shape[1])


    """
    TODO decommentare.
    """
    @staticmethod
    def saveModel(weights, eta, alfa, lambd, ntl, nhu, af, u_in, u_out, epochs, threshold, loss, i, final=False):
        now_m = datetime.now().isoformat()
        now = (now_m.rpartition(':')[0]).replace(":", "")
        # print(now)
        # folder = "models/Model_"+now+"/"
        i = str(i)

        if final:
            folder = "models/finals/Model" + i + "/weights/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            for k in weights:
                path = folder + k
                data = weights[k]
                print(path)
                np.savez(path, weights=data)
            folder = "models/finals/Model" + i + "/"

        else:
            folder = "models/Model_" + i + "/"
            if not os.path.exists(folder):
                os.makedirs(folder)

        path = folder + "hyperpar"
        np.savez(path, eta=eta, alfa=alfa, lambd=lambd, epochs=epochs, threshold=threshold, loss=loss)

        path = folder + "topology"
        np.savez(path, ntl=ntl, nhu=nhu, af=af, u_in=u_in, u_out=u_out)

        path = folder + "info.txt"
        with open(path, mode='w') as info_model:
            inf = "tot layer", ntl," \n- hidden units",nhu," \n- eta", eta, " \n- alfa", alfa," \n- lambda", lambd, " \n- activ func", af
            inf = str(inf)
            info_model.write('%s\n' % inf)


"""

    def plotError(self, epochs_plot, errors, ts_error):
        plt.plot(epochs_plot, errors, color="blue", label="training error")
        plt.plot(epochs_plot, ts_error, color="red", label="test error", linestyle="-.")
        plt.xlabel("epochs")
        plt.ylabel("error")
        plt.legend(loc='upper left', frameon=False)
        plt.show()

    def plot_accuracy(self, epochs_plot, accuracy, ts_accuracy):
        plt.plot(epochs_plot, accuracy, color="blue", label="accuracy TR")
        plt.plot(epochs_plot, ts_accuracy, color="red", label="accuracy TS", linestyle="-.")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend(loc='upper left', frameon=False)
        plt.show()
"""
