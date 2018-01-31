import numpy as np
from ML_CUP_dataset import ML_CUP_Dataset
import matplotlib.pyplot as plt

def activ_func(f_name, x):
    if f_name == 'sigmoid':
        a = 1
        return 1 / (1 + np.exp(- a * x))
    elif f_name == 'tanh':
        return np.tanh(x)
    elif f_name == 'softplus':
        return np.log(1 + np.exp(x))
    elif f_name == 'relu':
        y = x.copy()
        return y * (y > 0)
    elif f_name == 'linear':
        return x


def activation_function_derivative(f_name, x):
    if f_name == 'sigmoid':
        a = 1
        deriv = (1 / (1 + np.exp(- a * x))) * (1 - (1 / (1 + np.exp(- a * x))))
        return deriv
    if f_name == 'tanh':
        return 1 - (np.tanh(x)) ** 2
    if f_name == 'softplus':
        deriv = 1 / (1 + np.exp(- x))
        return deriv
    if f_name == 'relu':
        deriv = 1 / (1 + np.exp(- x))
        return deriv
    if f_name == 'linear':
        deriv = 1
        return deriv


def mean_euclidean_err(target_value, neurons_out, deriv=False):
    if deriv:
        err = mean_euclidean_err(target_value, neurons_out)
        return np.subtract(neurons_out, target_value) * (1 / err)
    res = np.subtract(neurons_out, target_value) ** 2  # matrice con righe = numero neuroni e colonne = numero di pattern  // è al contrario
    res = np.sum(res, axis=0)  # somma sulle righe ora res = vettore con 1 riga e colonne = numero di pattern. ogni elemento è (t-o)^2
    res = np.sqrt(res)
    res = np.sum(res, axis=0)  # somma sulle colonne
    return (res / target_value.shape[1])

def __main__():
    eta= 0.01
    alfa= 0.9
    lambd= 0.01
    epochs= 500

    filename = 'ML-CUP17-TR.csv'
    input_vect, target_vect = ML_CUP_Dataset.load_ML_dataset(filename)
    input_vect = input_vect[:, 0:2]
    target_vect = target_vect[:, 0:2]

    ''' creazione rete '''
    # input_layer - 10 units
    in_lay_net = input_vect
    in_lay_out = input_vect
    ones_row = np.ones((1, 2))
    in_lay_out = np.concatenate((in_lay_out, ones_row), axis=0)

    # hidden_layer - 2 units
    # pesi = unità input (10) + bias, 2 unità
    hid_lay_w = np.ones((11, 2), dtype=np.float64)
    hid_lay_w = hid_lay_w*0.7

    # output layer - 2 units
    out_lay_w = np.ones((3, 2), dtype=np.float64)
    out_lay_w = out_lay_w*0.7

    errors = []
    epochs_plot=[]
    #   TRAINING
    for epoch in range(150):
        # forward
        hid_lay_net = np.dot(hid_lay_w.T, in_lay_out)
        hid_lay_out = activ_func('relu', hid_lay_net)
        hid_lay_out = np.concatenate((hid_lay_out, ones_row), axis=0)

        out_lay_net = np.dot(out_lay_w.T, hid_lay_out)
        out_lay_out = activ_func('linear', out_lay_net)

        # backprop (out->hid)
        err_deriv = mean_euclidean_err(target_vect, out_lay_out, True)
        f_prime_out = activation_function_derivative('linear', out_lay_net)
        delta_out = err_deriv * f_prime_out

        out_lay_w_bp = np.delete(out_lay_w, -1, 0)
        f_prime_hid = activation_function_derivative('relu', hid_lay_net)
        delta_hid = np.dot(out_lay_w_bp, delta_out) * f_prime_hid

        #   (hid->out)
        hid_lay_dW = np.dot(in_lay_out, delta_hid.T)
        hid_lay_w = hid_lay_w - eta*hid_lay_dW

        out_lay_dW = np.dot(hid_lay_out, delta_out.T)
        out_lay_w = out_lay_w - eta*out_lay_dW

        err = mean_euclidean_err(target_vect, out_lay_out)
        errors.append(err)
        epochs_plot.append(epoch)
        #print (err)

    plt.plot(epochs_plot, errors, color="blue", label="training error")
    plt.xlabel("epochs")
    plt.ylabel("error")
    plt.legend(loc='upper right', frameon=False)
    plt.show()

    print(out_lay_out)









__main__()