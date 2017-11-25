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
        result = np.dot(x, weights)
        return result[0][0]

    def out_func(self):
        return NeuronUnit.sigmoid(self)

    def error_func(self):
        return y - NeuronUnit.out_func(self)

    #def delta(self):
    #    return -2 * NeuronUnit.error * NeuronUnit.sigmoid(NeuronUnit.net_func, True)

x = np.array([[1,2,3,4,5,6]])
y = 1
#weights = np.random.rand(1,6).T
weights = [[ 0.31852429],[ 0.8763374 ],[ 0.01358744],[ 0.55160163],[ 0.51340143],[ 0.27862657]]

neu = NeuronUnit(x,weights,y)
print(neu.error_func())



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
out = Layer(1, 0, inp.n_units+hid.n_units)
print(out.create_array_ids()'''

class InputLayer(Layer):
    a=1

class HiddenLayer(Layer):
    a = 1


class OutputLayer(Layer):
    a = 1

