# -*- coding: utf-8 -*-

# ==============================================================================
# E' una Rete Neurale di reti neurali neurali anch'esse
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================
from input_layer import InputLayer
#from save-load modello import Save, Load
import numpy as np


class NeuralNetwork:

    def __init__(self):
        self.input_layer = []
        self.hidden_layers = []
        self.output_layer = []
        #self.loss_func

    """
    Aggiunge un input layer alla rete,
    se il layer precedente viene sovrascritto stampa un warning
    """
    def add_input_layer(self, input_layer):
        if isinstance(input_layer, list):
            print('Input layer connected')
        else:
            print('WARNING:\tyou could have overwritten previous input layer, you\'re doomed man!')
        self.input_layer = input_layer

    """
    Aggiunge un hidden layer alla lista di hidden layers
    """
    def add_hidden_layer(self, hidden_layer):
        self.hidden_layers.append(hidden_layer)


    """
    Rimuove l'hidden layer alla posizione index della lista
    """
    def remove_hidden_layer(self, index):
        self.hidden_layers.pop(index)

    """
    Aggiunge un layer di output alla rete
    """
    def add_output_layer(self, output_layer):
        self.output_layer = output_layer

    '''
    Per specificare quale loss function utilizzare
    '''
    def define_loss(self, loss_function):
        self.loss_function = loss_function


    """
    Implementa la forward propagation calcolando l'output di ogni unità della
    Rete
    
    // aggiunti i vari .net e .output per poter richiamare le matrici dall'oggetto
    """
    def forward_propagation(self, input_vector):
        net = self.input_layer.net_function(input_vector)
        input_layer_out = self.input_layer.layer_output()
        self.input_layer.output = input_layer_out   # // aggiunto il.output

        if len(self.hidden_layers) <= 1:
            h_layer = self.hidden_layers[0]
            h_layer.net = h_layer.net_function(input_layer_out) # // aggiunto hl.net
            h_layer_out = h_layer.layer_output()
            h_layer.output = h_layer_out # // aggiunto hl.output

            self.output_layer.net = self.output_layer.net_function(h_layer_out) # // agg ol.net
            out_layer_out = self.output_layer.layer_output()
            self.output_layer.output = out_layer_out # // agg ol.out

        else:
            last_layer_out = input_layer_out  # necessario?
            for h_layer in self.hidden_layers:
                h_layer.net = h_layer.net_function(last_layer_out) # // agg hl.net
                last_layer_out = h_layer.layer_output()
                h_layer.output = last_layer_out  # // aggiunto hl.output

            self.output_layer.net = self.output_layer.net_function(last_layer_out)  # // agg ol.net
            out_layer_out = self.output_layer.layer_output()
            self.output_layer.output = out_layer_out  # // agg ol.out

        return self.output_layer.output

    def backpropagation(self, input_vector, target_value, err_func, eta):
        # delt = deriv(E/out) * f'(net)
        err_deriv = NeuralNetwork.mean_euclidean_err(target_value, self.output_layer.output, True)
        out_net = self.output_layer.net
        f_prime = self.output_layer.activation_function_derivative(out_net)
        delta_out = err_deriv * f_prime  # dovrebbe essere una matrice con colonne = numero di pattern // è pattern x n output units
        self.output_layer.deltas = delta_out
        prev_layer_delta = delta_out
        prev_layer_weights = self.output_layer.weights  # prev layer weights sono i pesi del layer precedente (quindi quello a destra quando si fa la backprop)
        for layer in reversed(self.hidden_layers):
            layer_net = layer.net
            f_prime = layer.activation_function_derivative(layer_net)
            ''' //
                tolta l'ultima riga dei weights, quella che - a regola - conteneva il bias 
                    guarda un po' se ha senso questa cosa e se era proprio quella riga
            '''
            prev_layer_weights = np.delete(prev_layer_weights, -1,0)
            transpose_weights = np.transpose(prev_layer_weights)    # // trasposta fatta a parte senza motivo
            #delta = np.dot(prev_layer_weights, prev_layer_delta) * f_prime
            #delta = np.dot(prev_layer_delta, np.transpose(prev_layer_weights)) * f_prime
            delta = np.dot(prev_layer_delta, transpose_weights) * f_prime
            layer.deltas = delta
            prev_layer_delta = delta
            prev_layer_weights = layer.weights

        # update weights
        ''' //
            prodotto tra delta e output del layer precedente e moltiplicazione con eta fatti a parte per comodità, non c'era particolare motivo
            le trasposte sono fatte giusto per farmi tornare la dimensione delle varie matrici
        '''
        dot_prod = np.dot(self.output_layer.deltas.T, self.hidden_layers[-1].output)    # //trasposta a caso
        eta_dot = eta * dot_prod
        self.output_layer.weights = self.output_layer.weights + eta_dot.T   # // altra trasposta a caso
        #i = -2
        for i in range(len(self.hidden_layers)):    # // prima era for layer in reverse(self.hidden_layers), così mi tornava meglio
            layer = self.hidden_layers[i]
            if i==0:
                dot_prod = np.dot(layer.deltas.T, self.input_layer.output)
                eta_dot = eta * dot_prod
                layer.weights = layer.weights + eta_dot.T
                ''' // weights è un dizionario per poter avere i pesi aggiornati raggiungibili dal nome del layer'''
                key = "hidden"+str(i)   # // in questo caso hidden0
                weights = ({key:layer.weights})
            else:
                dot_prod = np.dot(layer.deltas.T, self.hidden_layers[i-1].output)
                eta_dot = eta * dot_prod
                layer.weights = layer.weights + eta_dot.T
                key = "hidden" + str(i)     # // key: hidden1
                weights.update({key: layer.weights})
                #i = i-1

            # delta_w = layer.weights + np.dot(layer.deltas, input_vector.T)    // questo delta_w non l'avevo capito
            # weights = weights + eta * delta_w
        weights.update({'output': self.output_layer.weights})   # // key: output
        return err_func(target_value, self.output_layer.output), weights

    def train_network(self, input_vector, target_value, epochs, threshold, loss_func, eta): # // aggiunti i target_values
        loss = NeuralNetwork.mean_euclidean_err
        if loss_func == 'mean_euclidean':
            loss = NeuralNetwork.mean_euclidean_err
        elif loss_func == 'squared_err':
            loss = NeuralNetwork.squared_err
        else:
            print('WARNING:\t loss function unkown. Defaulted to mean_euclidean')
        for epoch in range(epochs):
            forward_prop = NeuralNetwork.forward_propagation(self, input_vector)
            back_prop = NeuralNetwork.backpropagation(self, input_vector, target_value, loss, eta)
            err = back_prop[0]
            weights = back_prop[1]
            if err < threshold:
                print('lavinia puzzecchia! trallallero taralli e vino')
                break
        keywords = ""
        for k in weights:
            key = str(k)
            keywords = keywords+key+"=weights['"+key+"'],"
        print(keywords[:-1])    # [:-1] per togliere l'ultima virgola
        print(weights['output'])
        '''
            da keywords in poi: costruzione di una stringa che abbia key = porzione dizionario weights
            per poter poi accedere ai singoli indici al momento del load
                np.savez("model.npz", hidden0=weights['hidden0'], hidden1=weights['hidden1'], output=weights['output'])
                funziona, ma volevo trovare un modo di non dovere scrivere a mano tutto
                
                np.savez("model.npz", keywords[:-1])  (vedi sotto) 
                si è rivelato fallimentare
        '''
        np.savez("model.npz", keywords[:-1])


        # to do ritornare il modello allenato sennò stiamo usando il computer come termosifone



    """
    Funzione di errore
    """
    @staticmethod
    def squared_err(target_value, neuron_out, deriv=False):
        if deriv:
            return -(target_value - neuron_out)  # segno meno? 
        return (target_value - neuron_out)**2

    """
    Calcola il MEE
    target_value = matrice che ha per righe i target e come colonne i pattern
    neurons_out = matrice che ha per righe gli output e come colonne i pattern
    divide il risultato per il numero di colonne di target value che dovrebbe
    """
    @staticmethod
    def mean_euclidean_err(target_value, neurons_out, deriv=False):
        if deriv:
            err = NeuralNetwork.mean_euclidean_err(target_value, neurons_out)
            return np.subtract(neurons_out, target_value) * (1 / err)
        res = np.subtract(neurons_out, target_value)**2  # matrice con righe = numero neuroni e colonne = numero di pattern  // è al contrario
        res = np.sqrt(res)
        res = np.sum(res, axis=0)  # somma sulle colonne. ora res = vettore con 1 riga e colonne = numero di pattern. ogni elemento è (t-o)^2
        res = np.sum(res, axis=0)  # somma sulle righe
        return (res / target_value.shape[1])
