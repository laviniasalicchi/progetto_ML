import numpy as np


def net_func(x, weights):
    result = np.dot(x, weights[0])
    return result
    
def activation_function(net):
    if net > 0:
        return 1
    else:
        return -1

def out(net, activation):
    return activation(net)
    
def error(target, output):
    return (target-output)

#funzione non usata
def update_weights(weights, train, target):
    target = input_vector[1]
    if (output == input_vector[1]):
        return weights
    else:
        weights_new = weights + eta * target * train
        return weights
        
def training(train, target, weights, eta):
    
    epoch_weights = weights
    
    for i in range(0, len(train[:,0])):
        #print(train[i])
        net = net_func(train[i], epoch_weights)
        print('net:\t',net)
        output = activation_function(net)
        if (activation_function(net) == target[i]):
            #do nothing
            a = 1
            #print('pesi immutati')
            #return weights
        else:
            weights_new = epoch_weights + eta * train[i] * (target[i] - output)
            epoch_weights = weights_new
            print('weights_new', weights_new)
    return epoch_weights

'''
-1 corrisponde all'uscita logica 0
è necessario usare -1 e 1 per evitare che i pesi siano sempre positivi
            '''
train = np.array([[0, 0, 1],
                  [1, 0, 1],
                  [0, 1, 1],
                  [1, 1, 1]])
target_and = np.array([-1, -1, -1, 1]) 
target_or = np.array([-1, 1, 1, 1])

#weights = np.random.rand(1,3)
weights = np.zeros((1,3), dtype=np.int)
eta = 0.02      #oracolo lavinia 

print(weights[0])
i = 0
while i < 1000: #in teoria sarebbe finchè i pesi non cambiano
    weights_new = training(train, target_and, weights, eta)
    if np.array_equal(weights, weights_new):
        print('training ended')
        break
    weigths = weights_new
    print('epoch:', i)
    i += 1


#test del modello
for i in range(0, len(train[:,0])):
    print('Expected:', 1 if target_and[i] > 0 else 0)
    net = net_func(train[i], weights)
    out = activation_function(net)
    print('Predicted:', 1 if out > 0 else 0 )
    


'''while True:
    weights_new = training(train, target_and, weights, eta)
    if (np.array_equal(weights, weights_new)):
        print('La smetti di presentarmi sempre la stessa tabella? La so a memoria, cazzo!')
        break
    weights = weights_new'''







