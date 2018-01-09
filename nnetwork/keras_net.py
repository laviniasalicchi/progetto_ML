# -*- coding: utf-8 -*-

import numpy as np
from monk_dataset import *
from keras.models import *
from keras.layers import *
import matplotlib.pyplot as plt


#   load dataset
monk_datas = MonkDataset.load_encode_monk('../datasets/monks-1.train')
monk_targets = monk_datas[0].T      # keras.utils.to_categorical(y, num_classes=None) --> potrebbe servire
monk_input = monk_datas[1].T
print("in keras - input", monk_input.shape)
print("in keras - target", monk_targets.shape)

#   creazione rete
neural_net = Sequential()
hidden_layer = Dense(5, input_dim=monk_input.shape[1], activation='sigmoid')
output_layer = Dense(1, activation='sigmoid')

neural_net.add(hidden_layer)
neural_net.add(output_layer)


#   configurazione learning process
#       loss = MSE
#       optimizer = stochastic gradient descent (learning rate, momentum, learning rate decay, Nesterov momentum)
neural_net.compile(loss='mean_squared_error', optimizer='sgd', metrics = ['accuracy'])


#   training della rete        fit(dati, targets, grandezza batch, epochs)
training = neural_net.fit(monk_input, monk_targets, batch_size=124, epochs=1000)

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



