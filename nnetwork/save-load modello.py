'''
np.savez(nome_file, keywords per ogni array/matrice)

np.load(file)
    .files per richiamare array/matrici usando le keywords
'''

import numpy as np
from tempfile import TemporaryFile

file = "modello.npz"

x = [[1,2], [3,4,5], [0, 0.5]]
y = ["boh", "cavallo", "cane", "scimmia", "ippopotamo", "gatto"]
a = np.matrix('1 2; 3 4')

np.savez(file, x=x, y=y, a=a)

#file.seek(0) # Only needed here to simulate closing & reopening file

npzfile = np.load(file)

npzfile.files

['y', 'x', 'a']

ics = npzfile['x']
ipsil = npzfile['y']
matr = npzfile['a']

print(ics[1][2], " - ", len(ics))
print(ipsil, " - ", len(ipsil))
print(matr, " - ", len(matr))
