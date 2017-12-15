# -*- coding: utf-8 -*-

# ==============================================================================
# E' un Perceptron, input layer
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================

import numpy as np
#from layer import Layer


class InputLayer(Layer):


    """
    unit_previous_layer = unità del layer precedente
    in questo caso = size dell'input vector
    """
    def create_weights(self, unit_previous_layer):
        self.weights = np.ones(self.n_units)

    """
    La net function nell'input non c'è
    ritorna l'input output_vector
    """
    def net_function(self, input_vector):
        self.net = input_vector
        return self.net

    """
    il layer output per l'input layer consiste nel vettore di input con l'aggiunta del bias

    """
    def layer_output(self):
        self.output_vector = self.net
        # aggiungiamo il bias
        self.output_vector = np.append(self.output_vector, [[1]], axis=0)
        return self.output_vector
