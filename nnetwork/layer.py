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


    """
    n_units = numero di unità del layers
    unit_previous_layer = unità del layer precedente 
    // non manca il bias così? Non dovremmo usare la matrice che è output del layer precedente?
    """
    def create_weights(self, unit_previous_layer):
        self.weights = np.random.rand(unit_previous_layer, self.n_units)

    """
    Calcola la funzione di rete come W trasposta per x
    ritorna un vettore di dimensione len(x)
    
    // tolta la trasposta, in quanto:
        input vector = matrice (pattern, feature+bias) OPPURE output del layer precedente = (pattern, unit_previous_layer+bias)
        weights = matrice (feature+bias, n_units) OPPURE (unit_previous_layer, n_units)
            colonne prima matrice = righe della seconda
    """
    def net_function(self, input_vector):
        #self.net = np.dot(np.transpose(self.weights), input_vector)
        self.net = np.dot(input_vector, self.weights)
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
        v_activation = np.vectorize(self.activation_function)  # applica la funzione di attivazione al vettore della net dei singoli neuroni
        self.output_vector = v_activation(self.net)
        # aggiungiamo il bias
        #self.output_vector = np.append(self.output_vector, [[1]], axis=0)
        #return self.output_vector
        self.output = np.ones((self.output_vector.shape[0], self.output_vector.shape[1] + 1))
        self.output[:, :-1] = self.output_vector
        return self.output

    """
    f_name: nome della funzione di attivazione da usare per i neuroni del
    layer
    """
    def set_activation_function(self, f_name):
        if f_name is 'sigmoid':
            self.activation_function = Layer.sigmoid
        elif f_name is 'tanh':
            self.activation_function = Layer.tanh
        else:
            self.activation_function = Layer.sigmoid
            print('WARNING:\tf_name not recognized. Using sigmoid as activation function')

    ''' //
        WARNING: la parte della sigmoid non ha subito modifiche
        usando la tanh ho visto che fare operazioni con self.activation_function senza argomento dava errore:
            unsupported operand type(s) for -: 'int' and 'function'
        vectorize pare non essere indispensabile
    '''

    def activation_function_derivative(self, x):
        if self.activation_function == Layer.sigmoid:
            #deriv = (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
            deriv = self.activation_function * (1 - self.activation_function)
            vectorized = np.vectorize(deriv)
            return vectorized(x)
        if self.activation_function == Layer.tanh:
            #deriv = 1 - self.activation_function**2
            #vectorized = np.vectorize(deriv)
            #return vectorized(x)
            return 1 - (self.activation_function(x))**2


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
