import numpy as np

x = np.array([[1,2,3,4,5,6]])       #matrice input

weights = np.random.rand(1,6).T     #T = trasposta
#y = np.array[(1,2,2,1,10)]
y = 3

# activation function sigmoid
# deriv per poter utilizzare in futuro la f'(net)
def sigmoid(x, deriv=False):
    a = 1                           #DA CAMBIARE
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def net(x,weights):
    return np.dot(x, weights)       #dot product tra vettore x e vettore pesi


def out(net,activation):
    return (activation(net))        # parametro activation cambia volta volta alla chiamata a seconda della funzione scelta


def error(target, output):
    return (target-output)**2

def delta(error,x,net):
    return (-2*x*error*sigmoid(net,True))


eta = 0.02                          # per ora eta basso--> online

#for i in range(10):

err = 5

y_train = out(net(x, weights), sigmoid)
print(type(net(x,weights)))

'''while err > 3:
    y_train = out(net(x, weights), sigmoid)
    print ("oooo", y_train)
    err = error(y, y_train)

    w_new = weights + eta*delta(err,x,net(x,weights))
    weights = w_new

    #print(weights)'''
