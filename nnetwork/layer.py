# -*- coding: utf-8 -*-

# ==============================================================================
# E' un Layer, che ti aspettavi?
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================
import numpy as np
import logging


class Layer:

    def __init__(self, n_units):
        self.weights = []
        self.net = []
        self.activation_function = []
        self.output = []  # output del layer: può essere una matrice // non lo usiamo mai?
        self.n_units = n_units
        self.deltas = []  # vettore di delta associato con il layer
        self.delta_rprop = []
        self.last_dW = 0
        self.sigmoid_slope = 1


    """
    n_units = numero di unità del layers
    unit_previous_layer = unità del layer precedente
    """
    def create_weights(self, unit_previous_layer):
        logger = logging.getLogger(__name__)

        self.weights = (np.random.rand(unit_previous_layer, self.n_units) * 1.4) - 0.7
        logger.debug('Weights shape: %s', str(self.weights.shape))

        ones_row = (np.random.rand(1, self.weights.shape[1]) * 1.4) - 0.7  # bias non inizializzato a uno ma random
        #ones_row = np.zeros((1, self.weights.shape[1]))
        logger.debug('Biases shape: %s', str(self.weights.shape))
        self.weights = np.concatenate((self.weights, ones_row), axis=0)

        self.delta_rprop = np.ones((self.weights.shape)) * 0.1

    """
    Inizializza i pesi usando il fan in.
    """
    def create_weights_fan_in(self, unit_previous_layer, fan_in):
        logger = logging.getLogger(__name__)
        interval = 1 / float(fan_in)

        self.weights =np.random.uniform(- interval, interval, (unit_previous_layer, self.n_units))

        ones_row = np.random.uniform(-interval, interval, (1, self.weights.shape[1]))  # bias non inizializzato a uno ma random
        #ones_row = np.zeros((1, self.weights.shape[1]))
        logger.debug('Biases shape: %s', str(self.weights.shape))
        self.weights = np.concatenate((self.weights, ones_row), axis=0)
        self.delta_rprop = np.ones((self.weights.shape)) * 0.1



    """
    Calcola la funzione di rete come W trasposta per x
    ritorna un vettore di dimensione len(x)
    """
    def net_function(self, input_vector):
        self.net = np.dot(np.transpose(self.weights), input_vector)
        #self.net = np.dot(input_vector, self.weights)
        return self.net

    """
    applica la funzione di attivazione a un vettore x
    ritorna un vettore x
    WARNING: RITORNA UNA MATRICE

    // per il bias:
        self.output inizializzato come matrice di 1 con stesse righe dell'output_vector, ma una colonna in più
        i valori di self.out per tutte le righe e per le colonne dalla prima alla penultima sono gli stessi di output_vector
        return matrice come output_vector, ma una colonna in più di 1
    """
    def layer_output(self):
        #print(self.activation_function)
        if self.activation_function == 'sigmoid':
            self.output = Layer.sigmoid(self.net, self.sigmoid_slope)
        elif self.activation_function == 'tanh':
            self.output = Layer.tanh(self.net)
        elif self.activation_function == 'softplus':
            self.output = Layer.softplus(self.net)
        elif self.activation_function == 'relu':
            self.output = Layer.relu(self.net)
<<<<<<< HEAD
        elif self.activation_function == 'linear':
            self.output = Layer.linear(self.net)
=======
>>>>>>> 20d112a21c0c6a3b36d0e8e8bcae68bd07c40b2a
        # aggiungiamo il bias

        ones_row = np.ones((1, self.output.shape[1]))
        self.output = np.concatenate((self.output, ones_row), axis=0)
        return self.output


    """
    f_name: nome della funzione di attivazione da usare per i neuroni del
    layer
    slope: parametro "a" della sigmoide, default = 1
    """
    def set_activation_function(self, f_name):
        if f_name == 'sigmoid':
            self.activation_function = 'sigmoid'
        elif f_name == 'tanh':
            self.activation_function = 'tanh'
        elif f_name == 'softplus':
            self.activation_function = 'softplus'
        elif f_name == 'relu':
            self.activation_function = 'relu'
<<<<<<< HEAD
        elif f_name == 'linear':
            self.activation_function = 'linear'
=======
>>>>>>> 20d112a21c0c6a3b36d0e8e8bcae68bd07c40b2a
        else:
            self.activation_function = 'sigmoid'
            print('WARNING:\tf_name not recognized. Using sigmoid as activation function')


    """
    a è la slope della sigmoide
    """
    def activation_function_derivative(self, x, a):
        if self.activation_function == 'sigmoid':
            deriv = (1 / (1 + np.exp(- a * x))) * (1 - (1 / (1 + np.exp(- a * x))))
            return deriv
        if self.activation_function == 'tanh':
            return 1 - (np.tanh(x))**2

    def activation_function_derivative(self, x):
        if self.activation_function == 'sigmoid':
            deriv = (1 / (1 + np.exp(- self.sigmoid_slope * x))) * (1 - (1 / (1 + np.exp(- self.sigmoid_slope * x))))
            return deriv
        if self.activation_function == 'tanh':
            return 1 - (np.tanh(x))**2
        if self.activation_function == 'softplus':
            deriv = 1 / (1 + np.exp(- x))
            return deriv
        if self.activation_function == 'relu':
            deriv = 1 / (1 + np.exp(- x))
<<<<<<< HEAD
            '''if (x>=0):
                deriv = 1
            else:
                deriv = 0'''
            return deriv
        if self.activation_function == 'linear':
            deriv = 1
=======
>>>>>>> 20d112a21c0c6a3b36d0e8e8bcae68bd07c40b2a
            return deriv

    def set_sigmoid_slope(self, slope):
        self.sigmoid_slope = slope


    """
    Sigmoid function
    """
    @staticmethod
    def sigmoid(x, slope):
        return 1 / (1 + np.exp(- slope * x))

    """
    Tanh function
    """
    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def softplus(x):
        return np.log(1 + np.exp(x))

    @staticmethod
    def relu(x):
        y = x.copy()
        return y * (y > 0)
<<<<<<< HEAD
        #return np.maximum(0, x)

    @staticmethod
    def linear(x):
        return x
=======
>>>>>>> 20d112a21c0c6a3b36d0e8e8bcae68bd07c40b2a
