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

def grid_search(input_vector, target_value, epochs, threshold, loss_func):
#def grid_search():
    '''n_hidden_units = np.arange(2, 11, 2)
    etas = np.arange(0.01, 0.51, 0.04)
    alfas = np.arange(0.5, 1.1, 0.2)
    lambds = np.arange(0.01, 0.11, 0.03)'''
    etas = [0.01, 0.05, 0.1, 0.3, 0.5]
    alfas = [0.5, 0.7, 0.9]
    lambds = [0.01, 0.04, 0.07, 0.1]
    i = 1
    for e in etas:
        for a in alfas:
            for l in lambds:
                #neural_net = NeuralNetwork.create_network(3, 17, 5, 1, 'sigmoid')
                #trained_net = neural_net.train_network(input_vector, target_value, epochs, threshold, loss_func, eta=e, alfa=a, lambd=l)

                acc = kfold_cv(input_vector, target_value, epochs, threshold, loss_func, eta=e, alfa=a, lambd=l)

                print(i, ")  eta:", e, " - alfa:", a, " - lambda:", l, "** ACCURACY:", acc)

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



def kfold_cv(input_vector, target_value, epochs, threshold, loss_func, eta, alfa, lambd):

    k = 4
    slice = int(input_vector.shape[1] / k)

    begin = 0
    errors_test = []
    for i in np.arange(0, input_vector.shape[1]+1, slice):
        print ("aiuto", i)
        if i != 0:
            print("CV n°",i)
            test = input_vector[:, begin:i]
            test_target_value = target_value[:, begin:i]
            train = np.delete(input_vector, np.s_[begin:i], 1)
            train_target_value = np.delete(target_value, np.s_[begin:i], 1)

            neural_net = NeuralNetwork.create_network(3, 17, 5, 1, 'sigmoid')

            trained_net = neural_net.train_network(train, train_target_value, epochs, threshold, loss_func, eta, alfa, lambd)
            neural_net_test = NeuralNetwork.create_network(3, 17, 5, 1, 'sigmoid')

            weights = trained_net[0]
            output_wei = trained_net[0]['output']
            neural_net_test.output_layer.weights = output_wei

            #print("PROVE")
            #print("neural net", neural_net.output_layer.weights)
            #print("trained_net[0]", trained_net[0]['output'])
            #print("neural net TEST", neural_net_test.output_layer.weights)

            trained_net[0].pop("output")

            l = 0
            for h in trained_net[0]:
                key = str(h)
                hidden_wei = trained_net[0][key]
                neural_net_test.hidden_layers[l].weights = hidden_wei
                l = l+1

            err_test = neural_net_test.test_network(test, test_target_value)
            #err_test = neural_net_test.accuracy(test, test_target_value)
            errors_test = np.append(errors_test, err_test)
            print("ACCURACY_in cv", err_test)
            begin = i

    mean = errors_test.mean()
    print(errors_test.shape)
    print(errors_test)
    print(mean)
    return mean
