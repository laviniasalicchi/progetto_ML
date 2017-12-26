'''
np.savez(nome_file, keywords per ogni array/matrice)

np.load(file)
    .files per richiamare array/matrici usando le keywords
'''

import numpy as np

file = "model.npz"

npzfile = np.load(file)

npzfile.files
['hidden0', 'hidden1', 'output']

out = npzfile['output']

print(out)