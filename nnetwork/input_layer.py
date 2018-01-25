# -*- coding: utf-8 -*-

# ==============================================================================
# E' un Perceptron, input layer
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================

import numpy as np
from layer import Layer


class InputLayer(Layer):

    def create_weights(self, unit_previous_layer):
        """
        Inizializza pesi del layer.

        unit_previous_layer = unità del layer precedente
        in questo caso = size dell'input vector
        """
        self.weights = np.ones(self.n_units)

    def net_function(self, input_vector):
        """
        Funzione di rete.

        La net function nell'input layer non c'è
        ritorna l'input vector.
        """
        self.net = input_vector
        return self.net

    def layer_output(self):
        """
        Output del Layer.

        Output = input + aggiunta bias
        """
        self.output_vector = self.net
        ones_row = np.ones((1, self.net.shape[1]))
        self.output_vector = np.concatenate((self.net, ones_row), axis=0)
        return self.output_vector
