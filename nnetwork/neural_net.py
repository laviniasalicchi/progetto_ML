# -*- coding: utf-8 -*-

# ==============================================================================
# E' una Rete Neurale di reti neurali neurali anch'esse
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================
from input_layer import InputLayer
#from save-load modello import Save, Load
import numpy as np
import os
from datetime import datetime
from layer import Layer
from input_layer import InputLayer
from hidden_layer import HiddenLayer
from output_layer import OutputLayer

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import re

import logging
import sys


class NeuralNetwork:

    def __init__(self):
        self.input_layer = []
        self.hidden_layers = []
        self.output_layer = []

    """
    Aggiunge un input layer alla rete,
    se il layer precedente viene sovrascritto stampa un warning
    """
    def add_input_layer(self, input_layer):
        if isinstance(input_layer, list):
            print('Input layer connected')
        else:
            print('WARNING:\tyou could have overwritten previous input layer, you\'re doomed man!')
        self.input_layer = input_layer

    """
    Aggiunge un hidden layer alla lista di hidden layers
    """
    def add_hidden_layer(self, hidden_layer):
        self.hidden_layers.append(hidden_layer)


    """
    Rimuove l'hidden layer alla posizione index della lista
    """
    def remove_hidden_layer(self, index):
        self.hidden_layers.pop(index)

    """
    Aggiunge un layer di output alla rete
    """
    def add_output_layer(self, output_layer):
        self.output_layer = output_layer

    '''
    Per specificare quale loss function utilizzare
    '''
    def define_loss(self, loss_function):
        self.loss_function = loss_function



    """
    Factory method per istanziare una rete.
    total_layers = numero totale di layers
    units_hidden = numero di unità per gli hidden layers
    units_out = unità di output
    units_in = unità input
    activ_func = funzione di attivazione dei layers
    """
    @staticmethod
    def create_network(total_layers, units_in, units_hidden, units_out, activ_func):
        neural_network = NeuralNetwork()
        hidden_num = total_layers - 2
        input_layer = InputLayer(units_in)
        input_layer.create_weights(units_in)
        input_layer.set_activation_function(activ_func)
        neural_network.add_input_layer(input_layer)
        hidden_l = HiddenLayer(units_hidden)
        hidden_l.create_weights(units_in)
        hidden_l.set_activation_function(activ_func)
        neural_network.add_hidden_layer(hidden_l)

        for i in range(1, hidden_num):
            hidden_l = HiddenLayer(units_hidden)
            hidden_l.create_weights(units_hidden)
            hidden_l.set_activation_function(activ_func)
            neural_network.add_hidden_layer(hidden_l)

        output_layer = OutputLayer(units_out)
        output_layer.create_weights(units_hidden)
        output_layer.set_activation_function(activ_func)
        neural_network.add_output_layer(output_layer)
        neural_network.print_net_info()
        return neural_network

    def predict(self, input_vector):
        return forward_propagation(self, input_vector)

    def print_net_info(self):
        print("**************** NETWORK ****************")
        print("Input Layer: " + str(self.input_layer.n_units) + ' units')
        i = 1
        for h_layer in self.hidden_layers:
            print("HiddenLayer " + str(i + 1) + ": " + str(self.hidden_layers[0].n_units) + " units")
            i = i + 1
        print("Output Layer: " + str(self.output_layer.n_units) + ' units')
        print("**************** NETWORK ***************")







    """
    Implementa la forward propagation calcolando l'output di ogni unità della
    Rete

    // aggiunti i vari .net e .output per poter richiamare le matrici dall'oggetto
    """
    def forward_propagation(self, input_vector):
        net = self.input_layer.net_function(input_vector)
        input_layer_out = self.input_layer.layer_output()
        self.input_layer.output = input_layer_out   # // aggiunto il.output
        #print('debug\t:inout_layer_out', self.input_layer.output.shape)


        if len(self.hidden_layers) <= 1:

            h_layer = self.hidden_layers[0]
            h_layer.net = h_layer.net_function(input_layer_out) # // aggiunto hl.net
            h_layer_out = h_layer.layer_output()
            h_layer.output = h_layer_out # // aggiunto hl.output

            self.output_layer.net = self.output_layer.net_function(h_layer_out) # // agg ol.net
            out_layer_out = self.output_layer.layer_output()
            self.output_layer.output = out_layer_out # // agg ol.out

        else:
            last_layer_out = input_layer_out  # necessario?
            for h_layer in self.hidden_layers:
                h_layer.net = h_layer.net_function(last_layer_out) # // agg hl.net
                last_layer_out = h_layer.layer_output()
                h_layer.output = last_layer_out  # // aggiunto hl.output

            self.output_layer.net = self.output_layer.net_function(last_layer_out)  # // agg ol.net
            out_layer_out = self.output_layer.layer_output()
            self.output_layer.output = out_layer_out  # // agg ol.out

        return self.output_layer.output


    """
    eta = learning rate
    alfa = momentum
    lambda = Tikhonov regularization
    """
    def backpropagation(self, input_vector, target_value, err_func, eta, alfa, lambd):
        logger = logging.getLogger(__name__)

        # delt = deriv(E/out) * f'(net)
        err_deriv = err_func(target_value, self.output_layer.output, True)
        logger.debug('Backprop: err_deriv.shape %s', str(err_deriv.shape))

        out_net = self.output_layer.net

        logger.debug("Backprop: out_net.shape %s", str(out_net.shape))

        f_prime = self.output_layer.activation_function_derivative(out_net)
        logger.debug("Backprop: f_prime shape %s", str(f_prime.shape))

        delta_out = err_deriv * f_prime  # dovrebbe essere una matrice con colonne = numero di pattern // è pattern x n output units
        self.output_layer.deltas = delta_out
        prev_layer_delta = delta_out
        prev_layer_weights = self.output_layer.weights  # prev layer weights sono i pesi del layer precedente (quindi quello a destra quando si fa la backprop)
        logger.debug("Backprop: delta_out.shape %s", str(delta_out.shape))

        for layer in reversed(self.hidden_layers):
            layer_net = layer.net
            logger.debug("Backprop: layer_net.shape %s", str(layer_net.shape))

            f_prime = layer.activation_function_derivative(layer_net)

            prev_layer_weights = np.delete(prev_layer_weights, -1, 0) # tolta la riga dei pesi del bias
            transpose_weights = np.transpose(prev_layer_weights)    # // trasposta fatta a parte senza motivo

            logger.debug("Backprop: prev_layer_weights.shape %s", str(prev_layer_weights.shape))
            logger.debug("Backprop: prev_layer_delta.shape %s", str(prev_layer_weights.shape))
            logger.debug("Backprop: f_prime.shape %s", str(f_prime.shape))

            delta = np.dot(prev_layer_weights, prev_layer_delta) * f_prime

            layer.deltas = delta

            logger.debug("Backprop: layer_deltas.shape %s", str(layer.deltas.shape))

            prev_layer_delta = delta
            prev_layer_weights = layer.weights

        # update weights
        logger.debug("Backprop: layer.weights.shape %s", str(layer.weights.shape))
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

            layer.weights = layer.weights - (eta * dW) + momentum - reg_term # +/- 2*lambda*layer.weights (per Tikhonov reg.)  //  + (alfa * prev_layer_delta)  (per momentum)
            #print("DW pre", layer.last_dW)
            layer.last_dW = - (eta * dW) + momentum
            #print("DW post", layer.last_dW)
            last_layer_out = layer.output

        return err_func(target_value, self.output_layer.output)

    def train_network(self, input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd): # // aggiunti i target_values
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
        weights_BT = {}     # // dizionario inizialmente vuoto per salvare il modello con l'errore più basso
        err_BT = 4.51536876901e+19  # // errore con valore inizialmente enorme, servirà per il backtracking
        for epoch in range(epochs):
            logger.info("Epoch %s", str(epoch))
            #print("EPOCH", epoch)
            forward_prop = NeuralNetwork.forward_propagation(self, input_vector)
            acc = NeuralNetwork.accuracy(self.output_layer.output, target_value)


            err = NeuralNetwork.backpropagation(self, input_vector, target_value, loss, eta, alfa, lambd)
            accuracy.append(acc)


            #print(err)
            errors.append(err)
            epochs_plot.append(epoch)


            """
            test per barra di progresso

            sys.stdout.write('\r')
            j = (epoch + 1 / epochs)
            sys.stdout.write("[%-20s] %d%%" % ('='*int(j), 100*j))
            sys.stdout.flush()"""


            # // creazione dizionario {nomelayer : pesi}
            for i in range(len(self.hidden_layers)):
                layer = self.hidden_layers[i]
                if i == 0:
                    #// weights è un dizionario per poter avere i pesi aggiornati raggiungibili dal nome del layer
                    key = "hidden"+str(i)
                    weights = ({key:layer.weights})
                else:
                    key = "hidden" + str(i)
                    weights.update({key:    layer.weights})
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

        #NeuralNetwork.saveModel(self, weights)
        #NeuralNetwork.saveModel(self, weights)
        logger.info("Saving %s", str(epoch))

        # // in ogni caso si plotta l'andamento dell'errore su tutte le epoch
        #NeuralNetwork.plotError(self, epochs_plot, errors)
        #NeuralNetwork.plot_accuracy(self, epochs_plot, accuracy)
        print("Accuracy;", accuracy[len(accuracy)-1])


        return weights, err

    def test_network(self, x, target_value):
        # solo forward + calcolo dell'errore
        NeuralNetwork.forward_propagation(self, x)
        err = NeuralNetwork.mean_euclidean_err(target_value, self.output_layer.output)
        accuracy = NeuralNetwork.accuracy(self.output_layer.output, target_value)
        return err, accuracy

    @staticmethod
    def accuracy(output_net, target):
        out_rounded = np.rint(output_net)
        result = np.where(out_rounded == target, 1, 0)
        result = np.mean(result)
        return result



    def test_existing_model(self, input, target, path):
        dirs = os.listdir(path)
        for dir in dirs:
            print(dir)
            dir_wei = path + dir + "/weights"
            print(dir_wei)
            wei_files = os.listdir(dir_wei)
            i = 0
            for file in wei_files:
                print("FILES", file)
                if file == 'output.npz':
                    print("output ok")
                    fileout = dir_wei + "/" + file
                    npzfile = np.load(fileout)
                    output_wei = npzfile['weights']
                    self.output_layer.weights = output_wei
                matchhidden = re.match(r'hidden([0-9]).npz', file)
                if matchhidden:
                    print("hidden ok")
                    print(file)
                    fileout = dir_wei + "/" + file
                    npzfile = np.load(fileout)
                    hidden_wei = npzfile['weights']
                    self.hidden_layers[i].weights = hidden_wei
                    i = i + 1
            NeuralNetwork.forward_propagation(self, input)
            acc = NeuralNetwork.accuracy(self.output_layer.output, target)
            print("Accuracy su test set", acc)




    """
    MSE - sicuramente sbagliato
        per regolarizzazione: aggiungere +lambda*(weights)**2
    """
    @staticmethod
    def mean_squared_err(target_value, neuron_out, deriv=False):
        if deriv:
            return - (np.subtract(target_value, neuron_out))
        res = np.subtract(target_value, neuron_out)**2
        res = np.sum(res, axis=1)
        return res/target_value.shape[1]


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
        res = np.subtract(neurons_out, target_value)**2  # matrice con righe = numero neuroni e colonne = numero di pattern  // è al contrario
        res = np.sqrt(res)
        res = np.sum(res, axis=0)  # somma sulle colonne. ora res = vettore con 1 riga e colonne = numero di pattern. ogni elemento è (t-o)^2
        res = np.sum(res, axis=0)  # somma sulle righe
        return (res / target_value.shape[1])


    """
    TODO decommentare.
    """
    @staticmethod
    def saveModel(weights, eta, alfa, lambd, i, accuracy, final=False):
        """now_m = datetime.now().isoformat()
        now = (now_m.rpartition(':')[0]).replace(":", "")
        #print(now)
        #folder = "models/Model_"+now+"/"
        i = str(i)

        if final:
            folder = "models/finals/Model"+i+"/weights/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            for k in weights:
                path = folder + k
                data = weights[k]
                print(path)
                np.savez(path, weights=data)
            folder = "models/finals/Model" + i + "/"

        else:
            folder = "models/Model_"+i+"/"
            if not os.path.exists(folder):
                os.makedirs(folder)

        path = folder+"eta"
        np.savez(path, eta = eta)

        path = folder + "alfa"
        np.savez(path, alfa = alfa)

        path = folder + "lambda"
        np.savez(path, lambd = lambd)

        path = folder + "accuracy"
        np.savez(path, accuracy = accuracy)
        """




    def plotError(self, epochs_plot, errors):
        plt.plot(epochs_plot, errors, color="blue", label="training error")
        plt.xlabel("epochs")
        plt.ylabel("error")
        plt.legend(loc='upper left', frameon=False)
        plt.show()

    def plot_accuracy(self, epochs_plot, accuracy):
        plt.plot(epochs_plot, accuracy, color="blue", label="accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend(loc='upper left', frameon=False)
        plt.show()
