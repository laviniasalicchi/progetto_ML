# ==============================================================================
# E' un Layer, che ti aspettavi?
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================

class Layer:
    layer_len = 0
    def __init__(self, n_units):
        self.n_units = n_units
        Layer.layer_len += n_units
