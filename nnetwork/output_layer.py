# -*- coding: utf-8 -*-

# ==============================================================================
# E' un Perceptron, che ti aspettavi?
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================
from layer import Layer
import numpy as np

class OutputLayer(Layer):

    def layer_output(self):
        Layer.layer_output(self)
        last_row_idx = self.output.shape[0] - 1
        self.output = np.delete(self.output, last_row_idx, axis=0)
        return self.output
