# -*- coding: utf-8 -*-

# ==============================================================================
# Oggetto che gestisce il training di una rete
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================

from neural_net import NeuralNetwork

class NeuralTrainer:

    def __init__(self, neural_net, **kwargs):
        self.net = neural_net
        self.eta = kwargs.get('eta', 0.01)
        self.lambd = kwargs.get('lambd', 0.00)
        self.epochs = kwargs.get('epochs', 100)
        self.alfa = kwargs.get('alfa', 0.00)
        self.loss = kwargs.get('loss', "mean_euclidean")
        self.threshold = kwargs.get('treshold', 0.00)
        self.rprop_delt0 = 0.1
        self.rprop_delt_max = 50

    def train_network(self, input_vector, target_value, input_test, target_test, save=False):

        nn_net = self.net

        if self.loss == 'mean_euclidean':
            loss = NeuralNetwork.mean_euclidean_err
        elif self.loss == 'mean_squared_err':
            loss = NeuralNetwork.mean_squared_err
        else:
            loss = NeuralNetwork.mean_squared_err
            print('WARNING:\t loss function unkown. Defaulted to mean_euclidean')
        errors = []
        accuracy = []
        epochs_plot = []
        ts_errors = []
        ts_accuracy = []
        weights_BT = {}  # // dizionario inizialmente vuoto per salvare il modello con l'errore più basso
        err_BT = 4.51536876901e+19  # // errore con valore inizialmente enorme, servirà per il backtracking
        for epoch in range(self.epochs):
            #logger.info("Epoch %s", str(epoch))

            output = nn_net.forward_propagation(input_vector)
            acc = NeuralNetwork.accuracy(output, target_value)

            err = nn_net.backpropagation(input_vector, target_value, loss, self.eta, self.alfa, self.lambd)
            accuracy.append(acc)
            errors.append(err)

            ts_err, ts_acc = nn_net.test_network(input_test, target_test)
            ts_accuracy.append(ts_acc)
            ts_errors.append(ts_err)

            epochs_plot.append(epoch)

            # // creazione dizionario {nomelayer : pesi}
            for i in range(len(nn_net.hidden_layers)):
                layer = nn_net.hidden_layers[i]
                if i == 0:
                    # // weights è un dizionario per poter avere i pesi aggiornati raggiungibili dal nome del layer
                    key = "hidden" + str(i)
                    weights = ({key: layer.weights})
                else:
                    key = "hidden" + str(i)
                    weights.update({key: layer.weights})
            weights.update({'output': nn_net.output_layer.weights})

            # // se l'errore scende sotto la soglia, si salva il modello che lo produce
            if err < self.threshold:
                print('lavinia puzzecchia! trallallero taralli e vino')
                #   NeuralNetwork.saveModel(self, weights)
                break
            # // se l'errore del modello di turno supera il precedente, si sovrascrive a weights il modello precedente
            # WARNING: l'errore può avere minimi locali, più avanti definiremo meglio questo if
            elif err > err_BT:
                weights = weights_BT
                #   NeuralNetwork.saveModel(self, weights)
            # // altrimenti, se l'errore continua a decrescere, si sovrascrive a weights_BT il modello corrente, si salva e si sovrascrive a error_BT l'errore del modello corrente
            else:
                weights_BT = weights
                err_BT = err

        if save:
            # todo parte di salvetaggio del modello

            NeuralNetwork.plotError(self, epochs_plot, errors, ts_errors)
            NeuralNetwork.plot_accuracy(self, epochs_plot, accuracy, ts_accuracy)
        return weights, err

    def _train_no_test(self, input_vector, target_value, save=False):
        nn_net = self.net

        if self.loss == 'mean_euclidean':
            loss = NeuralNetwork.mean_euclidean_err
        elif self.loss == 'mean_squared_err':
            loss = NeuralNetwork.mean_squared_err
        else:
            loss = NeuralNetwork.mean_squared_err
            print('WARNING:\t loss function unkown. Defaulted to mean_euclidean')
        errors = []
        accuracy = []
        epochs_plot = []
        ts_errors = []
        ts_accuracy = []
        weights_BT = {}  # // dizionario inizialmente vuoto per salvare il modello con l'errore più basso
        err_BT = 4.51536876901e+19  # // errore con valore inizialmente enorme, servirà per il backtracking
        for epoch in range(self.epochs):
            #logger.info("Epoch %s", str(epoch))

            output = nn_net.forward_propagation(input_vector)
            acc = NeuralNetwork.accuracy(output, target_value)
            err = nn_net.backpropagation(input_vector, target_value, loss, self.eta, self.alfa, self.lambd)
            accuracy.append(acc)
            errors.append(err)


            epochs_plot.append(epoch)

            # // creazione dizionario {nomelayer : pesi}
            for i in range(len(nn_net.hidden_layers)):
                layer = nn_net.hidden_layers[i]
                if i == 0:
                    # // weights è un dizionario per poter avere i pesi aggiornati raggiungibili dal nome del layer
                    key = "hidden" + str(i)
                    weights = ({key: layer.weights})
                else:
                    key = "hidden" + str(i)
                    weights.update({key: layer.weights})
            weights.update({'output': nn_net.output_layer.weights})

            # // se l'errore scende sotto la soglia, si salva il modello che lo produce
            if err < self.threshold:
                print('lavinia puzzecchia! trallallero taralli e vino')
                #   NeuralNetwork.saveModel(self, weights)
                break
            # // se l'errore del modello di turno supera il precedente, si sovrascrive a weights il modello precedente
            # WARNING: l'errore può avere minimi locali, più avanti definiremo meglio questo if
            elif err > err_BT:
                weights = weights_BT
                #   NeuralNetwork.saveModel(self, weights)
            # // altrimenti, se l'errore continua a decrescere, si sovrascrive a weights_BT il modello corrente, si salva e si sovrascrive a error_BT l'errore del modello corrente
            else:
                weights_BT = weights
                err_BT = err
        if save:
            NeuralNetwork.plotError(self, epochs_plot, errors, ts_errors)
            NeuralNetwork.plot_accuracy(self, epochs_plot, accuracy, ts_accuracy)
        print("Accuracy;", accuracy[len(accuracy) - 1])
        return weights, err

    def train_rprop(self, input_vector, target_value, input_test, target_test):
        nn_net = self.net
        loss = NeuralNetwork.mean_euclidean_err
        if loss_func == 'mean_euclidean':
            loss = NeuralNetwork.mean_euclidean_err
        elif loss_func == 'mean_squared_err':
            loss = NeuralNetwork.mean_squared_err
        else:
            print('WARNING:\t loss function unkown. Defaulted to mean_euclidean')
        errors = []
        accuracy = []
        epochs_plot = []
        ts_errors = []
        ts_accuracy = []
        err_BT = 4.51536876901e+19  # // errore con valore inizialmente enorme, servirà per il backtracking
        for epoch in range(epochs):
            logger.info("Epoch %s", str(epoch))
            output = nn_net.forward_propagation(input_vector)
            acc = NeuralNetwork.accuracy(output, target_value)

            err = nn_net.rprop(input_vector, target_value, loss, self.rprop_delt0, self.rprop_delt_max)
            accuracy.append(acc)
            errors.append(err)

            ts_err, ts_acc = nn_net.test_network(input_test, target_test)
            ts_accuracy.append(ts_acc)
            ts_errors.append(ts_err)

            epochs_plot.append(epoch)

        NeuralNetwork.plotError(self, epochs_plot, errors, ts_errors)
        NeuralNetwork.plot_accuracy(self, epochs_plot, accuracy, ts_accuracy)

    def train_rprop_no_test(self, input_vector, target_value, input_test, target_test):
        nn_net = self.net
        loss = NeuralNetwork.mean_euclidean_err
        if loss_func == 'mean_euclidean':
            loss = NeuralNetwork.mean_euclidean_err
        elif loss_func == 'mean_squared_err':
            loss = NeuralNetwork.mean_squared_err
        else:
            print('WARNING:\t loss function unkown. Defaulted to mean_euclidean')
        errors = []
        accuracy = []
        epochs_plot = []

        err_BT = 4.51536876901e+19  # // errore con valore inizialmente enorme, servirà per il backtracking
        for epoch in range(epochs):
            logger.info("Epoch %s", str(epoch))
            output = nn_net.forward_propagation(input_vector)
            acc = NeuralNetwork.accuracy(output, target_value)

            err = nn_net.rprop(input_vector, target_value, loss, self.rprop_delt0, self.rprop_delt_max)
            accuracy.append(acc)
            errors.append(err)
            epochs_plot.append(epoch)
