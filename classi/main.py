'''essendo momentaneamente incapace a richiamare classi da file esterni, piazzo tutto qui'''


import numpy as np

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

class Network:
    a=1


class NeuronUnit:
    a = 1


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



    def create_array_ids(self):
        array_units_id = list()
        for i in range(self.n_units):
            array_units_id.append(1+i+self.start_count)
        return array_units_id



'''inp = Layer(3, 0, 0)
hid = Layer(3, 0, inp.n_units)
out = Layer(1, 0, inp.n_units+hid.n_units)'''

class InputLayer(Layer):
    a=1

class HiddenLayer(Layer):
    a = 1


class OutputLayer(Layer):
    a = 1

