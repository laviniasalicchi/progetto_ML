import csv
import numpy as np

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

'''print("___________X__________")
print(x)
print("___________TARGET_X__________")
print(target_x)
print("___________TARGET_Y__________")
print(target_y)'''


''' --- fine estrazione/organizzazione dati da dataset ---
        
    --- inizio creazione layers + pesi --- '''

# n input units: dimensionalità degli input
# n_input_units = x.shape[1]                    #sarebbe quello giusto, per ora variabile toy
n_input_units = 2
n_hidden_units = 2
n_output_units = 1

#  PRIMA VERSIONE matrice per incrociare le unit e indicarne i collegamenti pesati
weights_matrix = np.empty([n_input_units + n_hidden_units + n_output_units, n_input_units + n_hidden_units + n_output_units])

# input_units sarà popolato volta volta allo scorrere dei pattern (conterrà le feature)
inputs = [2,6]                                  # !! provvisorio, poi verrà usato x !!

input_units = list()
hidden_units = list()
output_units = list()

# popolamento array DEGLI "ID" delle input units
# gli id partono da 1, quindi inp+1
for inp in range(n_input_units):
    input_units.append(inp+1)

print("input: ", input_units)

# popolamento array DEGLI "ID" delle hidden units
# gli id sono il proseguimento delle input units
for hide in range(n_hidden_units):
    hidden_units.append(len(input_units)+hide+1)

print("hide: ", hidden_units)

# popolamento array DEGLI "ID" delle output units
# gli id sono il proseguimento delle hidden units
for outs in range(n_output_units):
    output_units.append(len(input_units)+len(hidden_units)+outs+1)

print("out: ", output_units)

# concateno i tre array di layers
total_units = np.concatenate((input_units,hidden_units,output_units), axis=0)
print("total: ",total_units)

# array che "riassume" gli id per ogni layer --> potrebbe non servire a un cazzo, o forse si... boh
total_units_index = {'input': input_units, 'hide': hidden_units, 'output': output_units}
print(total_units_index)

# SECONDA VERSIONE matrice per incrociare le unit e indicarne i collegamenti pesati
weights_matrix = np.empty([len(total_units),len(total_units)])

# scorro la matrice e metto 0 al collegamento tra un'unità e se stessa, numero random tra 0 e 1 per gli altri
for row in range(weights_matrix.shape[0]):
    for col in range(weights_matrix.shape[1]):
        if (row==col):
            weights_matrix[row][col] = 0
        else:
            weights_matrix[row][col] = np.random.random_sample()


# creazione maschera - ma messa così è ancora soggetta alle modifiche dei pesi
weights_matrix_mask = np.empty([len(total_units),len(total_units)])

for row in range(weights_matrix.shape[0]):
    for col in range(weights_matrix.shape[1]):
        if (weights_matrix[row][col] == 0):
           weights_matrix_mask[row][col] = 0
        else:
            weights_matrix_mask[row][col] = 1


print(weights_matrix)
print(weights_matrix_mask)

'''
    to do:
        - trasferire tutto questo nelle classi
        - adattare per def delle classi 
'''