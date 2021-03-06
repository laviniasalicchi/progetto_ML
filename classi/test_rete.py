# -*- coding: utf-8 -*-

import numpy as np
#from classi import NeuronUnit
#import classi.OutputLayer
from layer import Layer



def __main__():
    ''' ---- parte per importare il dataset esterno ---- '''
    filename = 'ML-CUP17-TR.csv'
    raw_data = open(filename, 'r')

    data = np.loadtxt(raw_data, delimiter=",")

    x = np.empty([data.shape[0], data.shape[1] - 3])
    target_x = np.empty([data.shape[0], 1])
    target_y = np.empty([data.shape[0], 1])

    for i in range(0, len(data[:, 0])):
        #print("** ",i," **")
        k = 0
        for j in range(1,11):
            #print(j," - ", data[i][j])
            x[i][k] = data[i][j]
            k = k+1
        target_x[i][0] = data[i][11]
        target_y[i][0] = data[i][12]
    ''' ---- parte per importare il dataset esterno ---- '''


    layer = Layer(13)
    print(type(Layer.sigmoid))
    layer.set_activation_function('sigmoid')
    layer.create_weights(10)
    random_int = np.random.rand(10, 1)
    layer.net_function(random_int)
    output = layer.layer_output()
    print(output.shape)
    print(output)
    print('lavinia pappapero')

__main__()






"""class Network:

    def __init__(self,i_uni, h_uni, o_uni):
        self.input_units = i_uni
        self.hidden_units = h_uni
        self.output_units = o_uni

        #concateno i tre array di layers
        self.total_units = np.concatenate((self.input_units, self.hidden_units, self.output_units), axis=0)

        # array che "riassume" gli id per ogni layer --> potrebbe non servire a un cazzo, o forse si... boh
        self.total_units_index = {'input': self.input_units, 'hide': self.hidden_units, 'output': self.output_units}
        self.weights = Network.create_weights_matrix(self)

    def create_weights_matrix(self):
        # matrice per incrociare le unit e indicarne i collegamenti pesati
        weights_matrix = np.empty([len(self.total_units), len(self.total_units)])

        i=1
        # scorro la matrice e metto 0 al collegamento tra un'unità e se stessa, numero random tra 0 e 1 per gli altri
        for row in range(weights_matrix.shape[0]):
        return weights_matrix

    def create_weights_matrix_mask(self):
        # creazione maschera - ma messa così è ancora soggetta alle modifiche dei pesi
        weights_matrix_mask = np.empty([len(self.total_units), len(self.total_units)])

        weights_matrix = self.weights

        for row in range(weights_matrix.shape[0]):
            for col in range(weights_matrix.shape[1]):
                if (weights_matrix[row][col] == 0):
                    weights_matrix_mask[row][col] = 0
                else:
                    weights_matrix_mask[row][col] = 1
        return weights_matrix_mask

    def forward_propagation(self):
        for lay_dest in self.total_units_index:
            if lay_dest == "hide":
                for unit in self.total_units_index[lay_dest]:
                    print(unit," - ", self.weights[unit-1])
                    ''' PROBLEMA: ogni volta che chiamo self.weights riparte la creazione della matrice dei pesi
                                    con numeri random ogni volta'''






class NeuronUnit:
    def __init__(self, x, weights, y):
        self.x = x
        self.weights = weights
        self.y = y
        self.net = NeuronUnit.net_func(self)
        self.out = NeuronUnit.out_func(self)
        self.error = NeuronUnit.error_func(self)

    def sigmoid(self):
        a = 1
        #if (deriv == True):
        #    return self.x * (1 - self.x)
        return 1 / (1 + np.exp(-self.net))

    def net_func(self):
        result = np.dot(self.x, self.weights)
        return result[0][0]

    def out_func(self):
        return NeuronUnit.sigmoid(self)

    def error_func(self):
        return self.y - NeuronUnit.out_func(self)

    #def delta(self):
    #    return -2 * NeuronUnit.error * NeuronUnit.sigmoid(NeuronUnit.net_func, True)


class Layer:
    def __init__(self, n_units, layer_type, start_count):
        # n unità del layer
        self.n_units = n_units
        # tipo di layer (input, hidden, output) // forse non necessario
        self.layer_type = layer_type
        ''' per creazione array con gli id delle unità che appartengono al layer
                la conta progressiva degli id numerici dovrà essere il proseguimento dell'array del layer precedente
                    ---> lo start_count sarà la somma del numero di unità dei layer precedenti '''
        self.start_count = start_count
        self.array = Layer.create_array_ids(self)

    def create_array_ids(self):
        array_units_id = list()
        for i in range(self.n_units):
            array_units_id.append(1+i+self.start_count)
        return array_units_id



inp = Layer(3, 0, 0)
hid = Layer(2, 0, inp.n_units)
out = Layer(1, 0, inp.n_units+hid.n_units)


network = Network(inp.array,hid.array,out.array)
print(network.total_units_index)
print(network.total_units)
print(network.create_weights_matrix())
print(network.create_weights_matrix_mask())
print(network.forward_propagation())


class InputLayer(Layer):
    a=1


class HiddenLayer(Layer):
    a = 1


class OutputLayer(Layer):
    a = 1

#__main__()


'''
    da fare:
        - propagare in avanti l'input: l'output di un'unità deve diventare input della successiva
        - calcolare errore della rete
        - creare formule di backprop. personalizzate per ogni layer
            - aggiornare conseguentemente i pesi
        - al momento ogni unità è collegata con le altre. Bisogna trovare il modo di non inizializzare con tutti quei collegamenti
            - o lasciare così e implementare un pruning per sfoltire i collegamenti tra unità
'''
"""
