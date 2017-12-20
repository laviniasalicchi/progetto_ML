# ==============================================================================
# E' un Perceptron, che ti aspettavi?
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================
from layer import Layer
import numpy as np

class OutputLayer(Layer):
    def layer_output(self):
        v_activation = np.vectorize(self.activation_function)  # applica la funzione di attivazione al vettore della net dei singoli neuroni
        self.output_vector = v_activation(self.net)
        return self.output_vector