# -*- coding: utf-8 -*-

# ==============================================================================
# E' un Layer, che ti aspettavi?
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================
import numpy as np


class Layer:

    def __init__(self, n_units):
        self.weights = []
        self.net = []
        self.activation_function = []
        self.output = []  # output del layer: può essere una matrice // non lo usiamo mai?
        self.n_units = n_units
        self.deltas = []  # vettore di delta associato con il layer
        self.last_dW = 0


    """
    n_units = numero di unità del layers
    unit_previous_layer = unità del layer precedente
    // non manca il bias così? Non dovremmo usare la matrice che è output del layer precedente?
    """
    def create_weights(self, unit_previous_layer):
        self.weights = (np.random.rand(unit_previous_layer, self.n_units) * 1.4) - 0.7
        print('DEBUG:weights.shape:',self.weights.shape)
        ones_row = (np.random.rand(1, self.weights.shape[1]) * 1.4) - 0.7

        #ones_row = np.zeros((1, self.weights.shape[1]))
        print('DEBUG:ones.shape:', ones_row.shape)

        self.weights = np.concatenate((self.weights, ones_row), axis=0)

    """
    Calcola la funzione di rete come W trasposta per x
    ritorna un vettore di dimensione len(x)

    // tolta la trasposta, in quanto:
        input vector = matrice (pattern, feature+bias) OPPURE output del layer precedente = (pattern, unit_previous_layer+bias)
        weights = matrice (feature+bias, n_units) OPPURE (unit_previous_layer, n_units)
            colonne prima matrice = righe della seconda
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
        if self.activation_function is 'sigmoid':
            self.output = Layer.sigmoid(self.net)
        elif self.activation_function is 'tanh':
            self.output = Layer.tanh(self.net)
        # aggiungiamo il bias

        ones_row = np.ones((1, self.output.shape[1]))
        self.output = np.concatenate((self.output, ones_row), axis=0)
        return self.output
        #self.output = np.ones((self.output_vector.shape[0], self.output_vector.shape[1] + 1))
        #self.output[:, :-1] = self.output_vector
        #return self.output

    """
    f_name: nome della funzione di attivazione da usare per i neuroni del
    layer
    """
    def set_activation_function(self, f_name):
        if f_name is 'sigmoid':
            self.activation_function = 'sigmoid'
        elif f_name is 'tanh':
            self.activation_function = 'tanh'
        else:
            self.activation_function = 'sigmoid'
            print('WARNING:\tf_name not recognized. Using sigmoid as activation function')

    ''' //
        WARNING: la parte della sigmoid non ha subito modifiche
        usando la tanh ho visto che fare operazioni con self.activation_function senza argomento dava errore:
            unsupported operand type(s) for -: 'int' and 'function'
        vectorize pare non essere indispensabile
    '''

    def activation_function_derivative(self, x):
        if self.activation_function == 'sigmoid':
            deriv = (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
            return deriv
        if self.activation_function == 'tanh':
            #deriv = 1 - self.activation_function**2
            #vectorized = np.vectorize(deriv)
            #return vectorized(x)
            return 1 - (np.tanh(x))**2


    """
    Sigmoid function
    """
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    """
    Tanh function
    """
    @staticmethod
    def tanh(x):
        return np.tanh(x)
