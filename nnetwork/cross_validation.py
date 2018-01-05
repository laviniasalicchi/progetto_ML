# -*- coding: utf-8 -*-

# ==============================================================================
# E' una hidden layer
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================

import numpy as np

from layer import Layer
from input_layer import InputLayer
from hidden_layer import HiddenLayer
from output_layer import OutputLayer
from neural_net import NeuralNetwork
from monk_dataset import Monk_Dataset
from ML_CUP_dataset import ML_CUP_Dataset


'''
    MODEL SELECTION - CROSS-VALIDATION
        - importare dataset D
        - dividere il dataset D in K parti
        - per ogni valore dell'iperparametro teta:      // n hidden layer, n unità per layer, eta, ecc
            § per ogni parte D_k (e ogni controparte not_D_k):
                # training su not_D_k
                # test del modello risultante su D_k    // da salvare a parte
            § fare una media dei risultati dei test dai K modelli 
'''

'''filename = '../datasets/monks-1.train'
target_x = Monk_Dataset.load_encode_monk(filename)[0]
encoded_datas = Monk_Dataset.load_encode_monk(filename)[1]'''

# a= np.delete(a, np.s_[0:2], 1)


def kfold_cv(input_vector, target_value, epochs, threshold, loss_func, eta):

    k = 8
    slice = int(input_vector.shape[1] / k)

    begin = 0
    for i in np.arange(0, input_vector.shape[1]+1, slice):
        if i!=0:
            test = input_vector[:, begin:i]
            train = np.delete(input_vector, np.s_[begin:i], 1)
            train_target_value = np.delete(target_value, np.s_[begin:i], 1)

            input_layer = InputLayer(train.shape[0])
            input_layer.create_weights(train.shape[0])

            hidden_layer = HiddenLayer(5)
            hidden_layer.create_weights(input_layer.n_units)
            hidden_layer.set_activation_function('sigmoid')

            output_layer = OutputLayer(2)
            output_layer.create_weights(hidden_layer.n_units)
            output_layer.set_activation_function('sigmoid')

            neural_net = NeuralNetwork()
            neural_net.define_loss('mean_euclidean')
            neural_net.add_input_layer(input_layer)
            neural_net.add_hidden_layer(hidden_layer)
            neural_net.add_output_layer(output_layer)

            neural_net.forward_propagation(train)

            neural_net.train_network(train, train_target_value, 5, 10, 'mean_euclidean', 0.5)
            neural_net.test_network()

            begin = i