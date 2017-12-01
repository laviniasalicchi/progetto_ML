from random import seed
from random import random

# 1) **************** Initialize Network ****************

'''A network is organized into layers. 
    The input layer is really just a row from our training dataset. 
    The first real layer is the hidden layer. 
    This is followed by the output layer that has one neuron for each class value.

    We will organize layers as arrays of dictionaries and treat the whole network as an array of layers.'''

# numero inputs, unità hidden, unità output
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    ''' hidden layer = 
            n_hidden neuroni
                ognuno dei quali con n_input + 1 pesi --> uno per ogni input + bias'''
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    ''' output layer =
            n_output neuroni
                ognuno dei quali con n_hidden + 1 pesi --> ogni neurone dell'output layer 
                    ha una connessione pesata con ogni neurone dell'hidden'''
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


seed(1)
network = initialize_network(2, 1, 2)
for layer in network:
    print(layer)
    ''' run dello script:
            due layer:  hidden layer = UN neurone con 2 pesi (by input layer) + 1 peso by bias
                        output layer = DUE neuroni, ognuno con 1 peso + 1 peso bias '''


# 2) ************ Forward Propagate ************
    ''' 
        We can break forward propagation down into three parts:
            Neuron Activation
            Neuron Transfer
            Forward Propagation
    '''
# Neuron Activation (= net)
def activate(weights, inputs):
    # weights[-1] perché ogni layer p
    activation = weights[-1]
    print(weights[-1])
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# 3- Back Propagate Error.
# 4 - Train Network.
# 5 - Predict.
# 6- Seeds Dataset Case Study.