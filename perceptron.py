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

''' MODIFICATO IL DELTA: 
        - l'errore era (t-o)^2, ma nel delta serve solo (t-o), quindi ho messo la rad dell'errore
        - avevamo messo x, ma così moltiplicavamo tutto il vettore x ogni volta, 
            mentre nella formula dice di usare solo la feature in corrispondenza nel w che andavamo ad aggiornare
              quindi ho aggiunto i, indice del peso che aggiorniamo per selezionare l'elemento di x corrispondente'''
def delta(error,x,net,i):
    return (-2*x[0][i]*np.sqrt(error)*sigmoid(net,True))


eta = 0.02                          # per ora eta basso--> online


err = [5]

w_new = []                          # creo l'array di nuovi pesi


y_train = out(net(x, weights), sigmoid)
err = error(y, y_train[0])

sig = sigmoid(net(x,weights), deriv=True)
print (sig[0])

print("***** PESI VECCHI ****")
print(weights)
print("**********************")

'''contatore per gli indici dei pesi da modificare'''
i = 0

'''scorro il vettore dei pesi'''
for w in weights:
    print(i,")")
    '''aggiungo a w_new, volta volta, il risultato della formula con w_old, eta e delta
            passando a delta anche l'indice del peso che stiamo aggiornando'''
    w_new.append(weights[i] + eta * delta(err, x, net(x, weights), i))

    # roba random per la stampa
    net1 = net(x, weights)
    print("net ", net1)

    # altra roba random
    d = delta(err, x, net(x, weights), i)
    print("delta ", d)

    # idem
    print("w_old ", weights[i])
    print("w_new ", w_new[i])

    ''' sostituisco al vecchio peso quello nuovo '''
    weights[i] = w_new[i]

    print("----")
    i = i + 1

print("**** pesi nuovi ****")
print(weights)


''' stampando i delta, mi veniva un array la cui len era 1, ma di fatto conteneva 6 numeri, ex:
    [4  8   15  16  23  42]
    questo forse perchè aveva nella formula tutto il vettore x, anzichè la feature j-esima
    
    Nell'aggiornamento dei pesi, veniva fuori che w_new, che si riversava poi in weights, fosse un array di 6 elementi
    ma ognuno dei quali composto da 6 numeri, come il delta'''