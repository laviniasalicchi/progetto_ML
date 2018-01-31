# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
from ML_CUP_dataset import ML_CUP_Dataset
from keras.models import *
from keras.layers import *
from keras import losses
from keras.optimizers import *
from keras.regularizers import *

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plotError_noTS(epochs_plot, errors):
    plt.plot(epochs_plot, errors, color="blue", label="training error")
    plt.xlabel("epochs")
    plt.ylabel("error")
    plt.legend(loc='upper right', frameon=False)

    plt.show()


def activ_func(name, x):
    if name == 'sigmoid':
        a = 1
        y = x.copy()
        return 1 / (1 + np.exp(- a * y))
    elif name == 'relu':
        y = x.copy()
        return y * (y > 0)
    elif name == 'tanh':
        y = x.copy()
        return np.tanh(y)
    elif name == 'linear':
        y = x.copy()
        return y

def activ_func_derivative(name, x):
    if name == 'sigmoid':
        sig = activ_func('sigmoid', x.copy())
        deriv = sig * (1 - sig)
        return deriv
    elif name == 'relu':
        y = x.copy()
        y[y <= 0] = 0
        y[y > 0] = 1
        return y
    elif name == 'tanh':
        y = x.copy()
        deriv = 1 -(np.tanh(y)**2)
        return deriv
    elif name == 'linear':
        deriv = 1
        return 1



def mean_squared_err(target_value, neuron_out, deriv=False):
    if deriv:
        return - (np.subtract(target_value, neuron_out))
    res = np.subtract(target_value, neuron_out) ** 2
    res = np.sum(res, axis=0)
    res = np.sum(res, axis=0)
    return res / target_value.shape[1]

def mean_euclidean_err(target_value, neurons_out, deriv=False):
    if deriv:
        err = mean_euclidean_err(target_value, neurons_out)
        return np.subtract(neurons_out, target_value) * (1 / err)
    res = np.subtract(neurons_out, target_value) ** 2  # matrice con righe = numero neuroni e colonne = numero di pattern  // è al contrario
    res = np.sum(res, axis=0)  # somma sulle righe ora res = vettore con 1 riga e colonne = numero di pattern. ogni elemento è (t-o)^2
    res = np.sqrt(res)
    res = np.sum(res, axis=0)  # somma sulle colonne
    return  (res / target_value.shape[1])

def mean_euc_dist(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True)))

def _test_loss_func():
    target = np.array([[1,3,5], [2,3,5], [2, 5, 6]], dtype=np.float64)
    output = np.array([[1,3,5], [2,1,1], [1,1,1]], dtype=np.float64)
    print('\n----------- test loss: keras-------------', mean_euc_dist(output, target))
    print(mean_euc_dist(output, target))
    print('\------------ ours loss -----------------', mean_euclidean_err(target, output))
    print()




def __main__():
    filename = 'ML-CUP17-TR.csv'

    #---------------- keras --------------------------------------------------
    x = ML_CUP_Dataset.load_ML_dataset(filename)[0].T
    target_values = ML_CUP_Dataset.load_ML_dataset(filename)[1].T


    filename = 'ML-CUP17-TR.csv'
    input_vect, target_vect = ML_CUP_Dataset.load_ML_dataset(filename)
    #input_vect = input_vect[:, 0:2]
    #target_vect = target_vect[:, 0:2]

    '''
    #---------------------keras ----------------------------------------------
    neural_net_k = Sequential()
    hidden_layer = Dense(2, input_dim=x.shape[1], activation='sigmoid')
    output_layer = Dense(2, activation='linear')
    neural_net_k.add(hidden_layer)
    neural_net_k.add(output_layer)

    hidden_k_bias = hidden_layer.get_weights()[1].reshape(1, len(hidden_layer.get_weights()[1]))
    output_k_bias = output_layer.get_weights()[1].reshape(1, len(output_layer.get_weights()[1]))
    hidden_k_wei = np.concatenate((hidden_layer.get_weights()[0], hidden_k_bias))
    output_k_wei = np.concatenate((output_layer.get_weights()[0], output_k_bias))


    sgd_n = optimizers.SGD(lr=0.03, momentum=0.0, nesterov=False)
    neural_net_k.compile(loss=losses.mean_squared_error, optimizer=sgd_n, metrics = ['accuracy'])
    #training = neural_net_k.fit(x, target_values, batch_size=1016, epochs=500)

    plt.plot(training.history["loss"])
    plt.xlabel("epochs")
    plt.ylabel("error")
    plt.legend(loc='upper left', frameon=False)
    plt.show()'''


    '''params '''
    η = 0.01
    net_err_epochs = []
    epochs = 500
    hid_lay_af = 'sigmoid'
    out_lay_af = 'linear'
    hid_lay_units = 3
    out_lay_units = 2

    input_dim = input_vect.shape[0]
    output_dim = target_vect.shape[0]

    hid_lay_w = np.ones(shape=(input_dim, hid_lay_units)) * 0.7
    out_lay_w = np.ones(shape=(hid_lay_units, out_lay_units)) * 0.7

    in_lay_bias = np.zeros((1, hid_lay_units)).T
    out_lay_bias = np.zeros((1, out_lay_units)).T

    for e in range(epochs):
        # ############## FORWARD PROPAGATION #################################
        input_lay_out = input_vect.copy()
        #hid_lay_net = np.dot(hid_lay_w.T, input_lay_out) + in_lay_bias

        hid_lay_net = np.dot(hid_lay_w.T, input_lay_out,) + in_lay_bias
        print('\n\n================= in_lay bias ===================\n', in_lay_bias)


        print('\n\n================= hid_lay_net ===================\n', hid_lay_net)

        hid_lay_out = activ_func(hid_lay_af, hid_lay_net)

        out_lay_net = np.dot(out_lay_w.T, hid_lay_out) + out_lay_bias
        out_lay_out = activ_func(out_lay_af, out_lay_net)

        print('\n\n================= out_lay_out ===================\n', out_lay_out)

        net_error = mean_squared_err(target_vect, out_lay_out.copy())
        net_err_epochs.append(net_error)

        print('\n\n================= Error ===================\n', net_error)
        # ############### BACKPROPAGATION ##########################################

        err_deriv = mean_squared_err(target_vect.copy(), out_lay_out.copy(), True)
        f_prime_out = activ_func_derivative(out_lay_af, out_lay_net.copy())
        print('\n\n================= F_prime out ===================\n', f_prime_out)


        #delt = deriv(E/out) * f'(net)
        out_lay_delta = err_deriv * f_prime_out
        print('\n\n================= out_layer deltas ===================\n', out_lay_delta)

        f_prime_hid = activ_func_derivative(hid_lay_af, hid_lay_net.copy())
        print('\n\n================= f_prime_hid ===================\n', f_prime_hid)

        hid_lay_delta = np.dot(out_lay_w.copy(), out_lay_delta.copy()) * f_prime_hid

        print('\n\n================= hid_layer deltas ===================\n', out_lay_delta)

        Δw_hid = np.dot(input_lay_out, hid_lay_delta.T) * η

        print('\n\n================= Δw_hid  ===================\n', Δw_hid)

        Δw_out = np.dot(hid_lay_out, out_lay_delta.T) * η

        print('\n\n================= Δw_out  ===================\n', Δw_out)

        Δw_hid_bias = hid_lay_delta * η
        Δw_out_bias = out_lay_delta * η





        # weights update

        hid_lay_w = hid_lay_w - Δw_hid
        out_lay_w = out_lay_w - Δw_out

        in_lay_bias = in_lay_bias - Δw_hid_bias
        out_lay_bias = out_lay_bias - Δw_out_bias














    '''



    # creazione rete-
    # ------- input layer

    hid_lay_w = np.ones(shape=(10,2)) * 0.7
    hid_lay_w = np.concatenate((hid_lay_w, np.ones(shape=(1,2))), axis=0)
    hid_lay_w = hidden_k_wei # keras weights
    print('\n\n================= hid_lay_w ===================\n', hid_lay_w)

    out_lay_w = np.ones(shape=(2,2)) * 0.7
    out_lay_w = np.concatenate((out_lay_w, np.ones((1, 2))), axis=0)
    out_lay_w = output_k_wei #kears weights

    print('\n\n================= out_lay_w ===================\n', out_lay_w)

    for e in range(epochs):

        rand_patterns = input_vect[np.random.choice(input_vect.shape[0], 2, replace=False), :]

        print('----------------------___________---------------------------__________-----> epoch:', e)
        print()
        # ############## FORWARD PROPAGATION #################################

        input_lay_out = np.concatenate((input_vect.copy(), np.ones((1, input_vect.shape[1]))), axis=0)

        hid_lay_net = np.dot(hid_lay_w.T, input_lay_out)
        print('\n\n================= hid_lay_net ===================\n', hid_lay_net)

        hid_lay_out = activ_func(hid_lay_af, hid_lay_net)
        hid_lay_out = np.concatenate((hid_lay_out, np.ones((1, hid_lay_net.shape[1]))), axis=0)

        out_lay_net = np.dot(out_lay_w.T, hid_lay_out)
        #out_lay_net = np.concatenate((out_lay_net, np.ones((1, out_lay_net.shape[1]))),axis=0)
        out_lay_out = activ_func(out_lay_af, out_lay_net)

        print('\n\n================= out_lay_out ===================\n', out_lay_out)

        net_error = mean_squared_err(target_vect, out_lay_out.copy())
        net_err_epochs.append(net_error)

        print('\n\n================= Error ===================\n', net_error)

        # ############### BACKPROPAGATION ##########################################

        err_deriv = mean_euclidean_err(target_vect.copy(), out_lay_out.copy(), True)
        f_prime_out = activ_func_derivative(out_lay_af, out_lay_net.copy())
        print('\n\n================= F_prime out ===================\n', f_prime_out)


        #delt = deriv(E/out) * f'(net)
        out_lay_delta = err_deriv * f_prime_out
        print('\n\n================= out_layer deltas ===================\n', out_lay_delta)

        f_prime_hid = activ_func_derivative(hid_lay_af, hid_lay_net.copy())
        print('\n\n================= f_prime_hid ===================\n', f_prime_hid)


        out_lay_w_nobias = np.delete(out_lay_w.copy(), -1, 0)  # tolta la riga dei pesi del bias

        hid_lay_delta = np.dot(out_lay_w_nobias.T, out_lay_delta.copy()) * f_prime_hid

        print('\n\n================= hid_layer deltas ===================\n', out_lay_delta)

        Δw_hid = np.dot(input_lay_out, hid_lay_delta.T) * η

        print('\n\n================= Δw_hid  ===================\n', Δw_hid)

        Δw_out = np.dot(hid_lay_out, out_lay_delta.T) * η

        print('\n\n================= Δw_out  ===================\n', Δw_out)

        # weights update

        hid_lay_w = hid_lay_w - Δw_hid
        out_lay_w = out_lay_w - Δw_out

    '''

    last_net_error = mean_euclidean_err(target_vect, out_lay_out.copy())
    print('\n\n================= hid_lay_w Last===================\n', hid_lay_w)
    print('\n\n================= out_lay_w Last===================\n', out_lay_w)
    print('\n\n================= Final Error ===================\n', last_net_error)


    #_prova_backprop_new()


__main__()
