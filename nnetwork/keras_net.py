# -*- coding: utf-8 -*-

import numpy as np
from monk_dataset import *
from ML_CUP_dataset import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from neural_net import NeuralNetwork
from keras.regularizers import *
from trainer import *

filename = 'ML-CUP17-TR.csv'
x = ML_CUP_Dataset.load_ML_dataset(filename)[0].T
target_values = ML_CUP_Dataset.load_ML_dataset(filename)[1].T

neural_net_k = Sequential()
hidden_layer = Dense(10, input_dim=x.shape[1], activation='sigmoid')
output_layer = Dense(2, activation='linear')
neural_net_k.add(hidden_layer)
neural_net_k.add(output_layer)

def mean_euc_dist(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True)))


sgd_n = optimizers.SGD(lr=0.03, momentum=0.0, nesterov=False)
neural_net_k.compile(loss=mean_euc_dist, optimizer=sgd_n, metrics = ['accuracy'])
training = neural_net_k.fit(x, target_values, batch_size=1016, epochs=1000)


#   creazione rete
neural_net_k = Sequential()
hidden_layer = Dense(3, input_dim=x.shape[1], activation='relu')
output_layer = Dense(2, activation='linear')

neural_net_k.add(hidden_layer)
neural_net_k.add(output_layer)



#   prove random

hidden_k_bias = hidden_layer.get_weights()[1].reshape(1, len(hidden_layer.get_weights()[1]))
output_k_bias = output_layer.get_weights()[1].reshape(1, len(output_layer.get_weights()[1]))
hidden_k_wei = np.concatenate((hidden_layer.get_weights()[0], hidden_k_bias))
output_k_wei = np.concatenate((output_layer.get_weights()[0], output_k_bias))

unit_lay = [10, 3, 2]
af = ['linear','relu', 'linear']
neural_net = NeuralNetwork.create_advanced_net(3, unit_lay, af, "no")
neural_net.hidden_layers[0].weights = hidden_k_wei
neural_net.output_layer.weights = output_k_wei

input_vect, target_vect = ML_CUP_Dataset.load_ML_dataset(filename)

train_par = {
        'eta': 0.03,
        'alfa': 0.9,
        'lambd': 0.01,
        'epochs': 1000,
        'threshold': 0.0,
        'loss': 'mean_euclidean'
    }
trainer = NeuralTrainer(neural_net, **train_par)
err_net = trainer._train_no_test(input_vect, target_vect, save=True)
print("errore net", err_net[1])



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
