# -*- coding: utf-8 -*-

import numpy as np
from monk_dataset import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from neural_net import NeuralNetwork


#   load dataset
monk_datas = MonkDataset.load_encode_monk('../datasets/monks-1.train')
monk_targets = monk_datas[0].T      # keras.utils.to_categorical(y, num_classes=None) --> potrebbe servire
monk_input = monk_datas[1].T
print("in keras - input", monk_input.shape)
print("in keras - target", monk_targets.shape)

#   creazione rete
neural_net_k = Sequential()
hidden_layer = Dense(5, input_dim=monk_input.shape[1], activation='sigmoid')
output_layer = Dense(1, activation='sigmoid')

neural_net_k.add(hidden_layer)
neural_net_k.add(output_layer)



#   prove random

hidden_k_bias = hidden_layer.get_weights()[1].reshape(1,len(hidden_layer.get_weights()[1]))
output_k_bias = output_layer.get_weights()[1].reshape(1,len(output_layer.get_weights()[1]))
hidden_k_wei = np.concatenate((hidden_layer.get_weights()[0], hidden_k_bias))
output_k_wei = np.concatenate((output_layer.get_weights()[0], output_k_bias))

neural_net = NeuralNetwork.create_network(3, 17, 5, 1, 'sigmoid')
neural_net.hidden_layers[0].weights = hidden_k_wei
neural_net.output_layer.weights = output_k_wei

monk_targets_n = monk_datas[0]
monk_input_n = monk_datas[1]
err_net = neural_net.train_network(monk_input_n, monk_targets_n, 1000, 0.00001, 'squared_err', 0.1, alfa=0.0, lambd=0.0)


#   configurazione learning process
#       loss = MSE
#       optimizer = stochastic gradient descent (learning rate, momentum, learning rate decay, Nesterov momentum)
sgd_n = optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
neural_net_k.compile(loss='mean_squared_error', optimizer=sgd_n, metrics = ['accuracy'])


#   training della rete        fit(dati, targets, grandezza batch, epochs)
training = neural_net_k.fit(monk_input, monk_targets, batch_size=124, epochs=8000)

print("ERRORE NET", err_net)

# plots
# error
plt.plot(training.history["loss"])
plt.xlabel("epochs")
plt.ylabel("error")
plt.legend(loc='upper left', frameon=False)
plt.show()

# accuracy
plt.plot(training.history["acc"])
plt.title("accuracy")
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.show()
