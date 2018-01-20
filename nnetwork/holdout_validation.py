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
    HOLD-OUT VALIDATION
        1) dividere il dataset in 
            training set
            validation set
        2) per ogni combo di iperparametri:
            - training su TR
            - "test" su validation
        3) prendo la combo con accuracy più alta
        
        4) retraining su TR+VL
        
        5) valutare modello su un TS esterno
'''

def hold_out(input_vector, target_value, input_test, target_test, epochs, threshold, loss_func):
    bound = int(np.rint((input_vector.shape[1]/100)*60))
    training_set = input_vector[:, 0:bound]
    target_training = target_value[:, 0:bound]
    valid_set = input_vector[:, bound:input_vector.shape[1]]
    target_valid = target_value[:, bound:input_vector.shape[1]]

    etas = [0.01, 0.05, 0.1, 0.3, 0.5]
    alfas = [0.5, 0.7, 0.9]
    lambds = [0.01, 0.04, 0.07, 0.1]

    neural_net = NeuralNetwork.create_network(3, 17, 5, 1, 'sigmoid')
    i = 1
    for e in etas:
        for a in alfas:
            for l in lambds:
                trained = neural_net.train_network(training_set, target_training, epochs, threshold, loss_func, eta=e, alfa=a, lambd=l)
                valid = neural_net.test_network(valid_set, target_valid)
                err = valid[0]
                acc = valid[1]

                print(i, ")  eta:", e, " - alfa:", a, " - lambda:", l, "errore su training set:",trained[1]," errore sul valid:",err,"** ACCURACY su valid:", acc)

                NeuralNetwork.saveModel(0, e, a, l, i, acc)
                i=i+1
    accuracies = []
    for i in range(len(etas) * len(alfas) * len(lambds)):
        res = str(i + 1)
        file = "models/Model_" + res + "/accuracy.npz"
        accfile = np.load(file)
        acc = accfile['accuracy']
        accuracies = np.append(accuracies,acc)
    maxind = str(np.argmax(accuracies)+1)
    print("TOTAL ACCURACIES", accuracies)

    path = "models/Model_" + maxind + "/eta.npz"
    file = np.load(path)
    eta = file['eta']

    path = "models/Model_" + maxind + "/alfa.npz"
    file = np.load(path)
    alfa = file['alfa']

    path = "models/Model_" + maxind + "/lambda.npz"
    file = np.load(path)
    lambd = file['lambd']

    neural_net = NeuralNetwork.create_network(3, 17, 5, 1, 'sigmoid')
    train = neural_net.train_network(input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd, final=True)
    weights = train[0]
    error = train[1]
    accuracy = neural_net.accuracy(neural_net.output_layer.output, target_value)
    neural_net.saveModel(weights, eta, alfa, lambd, maxind, accuracy, final=True)

    neural_net.test_existing_model(input_test, target_test)