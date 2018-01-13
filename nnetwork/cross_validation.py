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
from monk_dataset import MonkDataset
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
            
            
         ### IPERPARAMETRI ###
            - range valori pesi al momento dell'inizializzazione
            - (se online/batch)
            - eta
                § momentum
            - (n epoch / stopping criteria)
            - lambda
            - n unità   
'''

'''filename = '../datasets/monks-1.train'
target_x = Monk_Dataset.load_encode_monk(filename)[0]
encoded_datas = Monk_Dataset.load_encode_monk(filename)[1]'''

# a= np.delete(a, np.s_[0:2], 1)

#def grid_search(input_vector, target_value, epochs, threshold, loss_func):
def grid_search():
    eta = np.arange(0.01, 0.5, 0.05)
    n_hidden_units = np.arange(2, 11, 2)
    for e in eta:
        for h in n_hidden_units:
            a=1
            '''
               si richiama volta volta kfold_cv 
            '''


def kfold_cv(input_vector, target_value, epochs, threshold, loss_func, eta):

    k = 8
    slice = int(input_vector.shape[1] / k)

    begin = 0
    errors_test = []
    for i in np.arange(0, input_vector.shape[1]+1, slice):
        print ("aiuto", i)
        if i!=0:
            print("CV n°",i)
            test = input_vector[:, begin:i]
            test_target_value = target_value[:, begin:i]
            train = np.delete(input_vector, np.s_[begin:i], 1)
            train_target_value = np.delete(target_value, np.s_[begin:i], 1)
            print("begin", begin, " - i", i)
            print("shapes:\ntest", test.shape, "\ntarget test",test_target_value.shape, "\ntrain", train.shape,"\ntarget train", train_target_value.shape)

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
            #neural_net.forward_propagation(train)

            trained_net = neural_net.train_network(train, train_target_value, epochs, threshold, loss_func, eta)
            print(trained_net[0])
            print(trained_net[1])
            print(trained_net[2].weights.shape)

            input_layer = InputLayer(test.shape[0])
            input_layer.create_weights(test.shape[0])

            for h in trained_net[1]:
                hidden_layer = HiddenLayer(5)
                hidden_layer.weights = h.weights
                hidden_layer.set_activation_function('sigmoid')

            output_layer = OutputLayer(2)
            output_layer.weights = trained_net[2].weights
            output_layer.set_activation_function('sigmoid')

            neural_net_test = NeuralNetwork()
            neural_net_test.define_loss('mean_euclidean')
            neural_net_test.add_input_layer(input_layer)
            neural_net_test.add_hidden_layer(hidden_layer)
            neural_net_test.add_output_layer(output_layer)

            err_test = neural_net_test.test_network(test, test_target_value)
            errors_test = np.append(errors_test, err_test)
            print("errore test", err_test.shape)
            mean = errors_test.mean()

            begin = i

    print(errors_test.shape)
    print(mean)