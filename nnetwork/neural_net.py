# -*- coding: utf-8 -*-

# ==============================================================================
# E' una Rete Neurale di reti neurali neurali anch'esse
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================
from input_layer import InputLayer
#from save-load modello import Save, Load
import numpy as np


class NeuralNetwork:

    def __init__(self):
        self.input_layer = []
        self.hidden_layers = []
        self.output_layer = []
        #self.loss_func

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
    Implementa la forward propagation calcolando l'output di ogni unità della
    Rete
    """
    def forward_propagation(self, input_vector):
        net = self.input_layer.net_function(input_vector)
        input_layer_out = self.input_layer.layer_output()
        if len(self.hidden_layers) <= 1:
            h_layer = self.hidden_layers[0]
            h_layer.net_function(input_layer_out)
            h_layer_out = h_layer.layer_output()
            self.output_layer.net_function(h_layer_out)
            out_layer_out = self.output_layer.layer_output()
        else:
            last_layer_out = input_layer_out  # necessario?
            for h_layer in self.hidden_layers:
                h_layer.net_function(last_layer_out)
                last_layer_out = h_layer.layer_output()
            self.output_layer.net_function(last_layer_out)
            out_layer_out = self.output_layer.layer_output()

    def backpropagation(self, input_vector, err_func, eta):
        layers = list()
        list.append(self.hidden_layers)
        # delt = deriv(E/out) * f'(net)
        '''
            - da dove prendiamo il target value da passare alla funzione di errore?
        '''
        err_deriv = NeuralNetwork.mean_euclidean_err(target_value, self.output_layer.output, True)
        out_net = self.output_layer.net
        f_prime = self.output_layer.activation_function_derivative(out_net)
        delta_out = err_deriv * f_prime  # dovrebbe essere una matrice con colonne = numero di pattern
        self.output_layer.deltas = delta_out
        prev_layer_delta = delta_out
        prev_layer_weights = self.output_layer.weights  # prev layer weights sono i pesi del layer precedente (quindi quello a destra quando si fa la backprop)
        for layer in reversed(layers):
            layer_net = layer.net
            f_prime = layer.activation_function_derivative(layer_net)
            delta = np.dot(prev_layer_weights, prev_layer_delta) * f_prime
            layer.deltas = delta
            prev_layer_delta = delta
            prev_layer_weights = layer.weights

        # update weights
        for layer in reversed(layers):
            delta_w = layer.weights + np.dot(layer.deltas, input_vector.T)
            weights = weights + eta * delta_w

        return err_func(target_value, self.output_layer.output), weights


    def train_network(self, input_vector, epochs, threshold, loss_func, eta):
        loss = NeuralNetwork.mean_euclidean_err
        if loss_func == 'mean_euclidean':
            loss = NeuralNetwork.mean_euclidean_err
        elif loss_func == 'squared_err':
            loss = NeuralNetwork.squared_err
        else:
            print('WARNING:\t loss function unkown. Defaulted to mean_euclidean')
        for epoch in epochs:
            NeuralNetwork.forward_propagation(input_vector)
            back_prop = NeuralNetwork.backpropagation(input_vector, loss, eta)
            err = back_prop[0]
            weights = back_prop[1]
            if err < threshold:
                print('lavinia puzzecchia! trallallero taralli e vino')
                break
        np.savez("model.npz", weights=weights)

        # todo ritornare il modello allenato sennò stiamo usando il computer come termosifone 



    """
    Funzione di errore
    """
    @staticmethod
    def squared_err(target_value, neuron_out, deriv=False):
        if deriv:
            return -(target_value - neuron_out)  # segno meno? 
        return (target_value - neuron_out)**2

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
        res = (target_value - neurons_out)**2  # matrice con righe = numero neuroni e colonne = numero di pattern
        res = np.sqrt(res)
        res = np.sum(res, axis=0)  # somma sulle colonne. ora res = vettore con 1 riga e colonne = numero di pattern. ogni elemento è (t-o)^2
        res = np.sum(res, axis=0)  # somma sulle righe
        return (res / target_value.shape[1])
