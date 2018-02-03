import numpy as np

from layer import Layer
from input_layer import InputLayer
from hidden_layer import HiddenLayer
from output_layer import OutputLayer
from neural_net import NeuralNetwork
from monk_dataset import MonkDataset
from ML_CUP_dataset import ML_CUP_Dataset
from grid_searcher import *
import os

def __main()__:





def grid_tests(input_vector, target_value, epochs, threshold, loss_func):
    bound = int(np.rint((input_vector.shape[1]/100)*70))
    training_set = input_vector[:, 0:bound]
    target_training = target_value[:, 0:bound]
    valid_set = input_vector[:, bound:input_vector.shape[1]]
    target_valid = target_value[:, bound:input_vector.shape[1]]

    grid_layers =[[10,5,2],[10,10,2],[10,5,5,2]]
    #af = ['relu','relu', 'relu', 'linear']
    af = ['relu']

    etas = [0.01, 0.05, 0.1]
    alfas = [0.7, 0.9]
    models = []
    i = 0
    for unit_lay in grid_layers:
        for e in etas:
            for a in alfas:
                folder = str("models_CUP/RELAZIONE/"+str(i))
                train_par = {
                    'eta': e,
                    'alfa': a,
                    'lambd': 0.01,
                    'epochs': 200,
                    'threshold': 0.0,
                    'loss': 'mean_euclidean'
                }
                file = folder+"info.txt"
                print(i,")""eta", e,"alfa", a,"lambda", 0.01, "arr layers", unit_lay)
                with open(file, mode='w') as infomodel:
                    inf = str("eta:"+str(e)+" - alfa:"+str(a)+" - lambda: 0.01")
                    infomodel.write('%s\n' % inf)

                tot_lay = len(unit_lay)
                ac_f = af * (len(unit_lay) - 1)
                ac_f.append('linear')

                neural_net = NeuralNetwork.create_advanced_net(tot_lay, unit_lay, ac_f, "xavier")
                trainer = NeuralTrainer(neural_net, **train_par)
                trainer.train_network(training_set, target_training, valid_set, target_valid, folder, save=True)
                i=i+1

__main()__