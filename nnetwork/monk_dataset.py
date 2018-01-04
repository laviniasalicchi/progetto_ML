# -*- coding: utf-8 -*-

# ==============================================================================
# E' una hidden layer
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================
'''
1. class: 0, 1
2. a1: 1, 2, 3
3. a2: 1, 2, 3
4. a3: 1, 2
5. a4: 1, 2, 3
6. a5: 1, 2, 3, 4
7. a6: 1, 2
8. Id: (A unique symbol for each instance)

0.  a1_1
1.  a2_1
2.  a3_1
3.  a4_1
4.  a5_1
5.  a6_1
6.  a1_2
7.  a1_3
8.  a2_2
9.  a2_3
10. a3_2
11. a4_2
12. a4_3
13. a5_2
14. a5_3
15. a5_4
16. a6_2
'''

import numpy as np
import csv

class Monk_Dataset:


    #filename = '../datasets/monks-1.train'

    '''
    Funzione per leggere e fare l'encoding del monk,
    ritorna un vettore con i valori della classe,
    e una matrice con i valori degli attributi
    salva su disco la versione codificata del file
    '''
    def load_encode_monk(filename):
        fields = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']
        print(len(fields))
        raw_data = open(filename, 'r')
        data = np.loadtxt(raw_data, delimiter=" ", dtype='str')
        data = np.delete(data, 0, axis=1)  # siccome è in un formato pessimo occorre togliere la prima colonna in quanto legge gli spazi come se fossero valori
        data = np.delete(data, data.shape[1] - 1, axis=1)
        target_x = np.empty((data.shape[0], 1))
        encoded_datas = np.zeros((data.shape[0], 17))
        counter = 0  # righe

        for counter in range(data.shape[0]):
            target_x[counter][0] = data[counter][0]

            for i in range(1, 7):
                # attributo a1
                if i is 1:
                    print(' i is ', i)
                    if np.equal(int(data[counter][i]), 1):
                        print('a1_value=1', int(data[counter][i]))
                        encoded_datas[counter][0] = 1
                        encoded_datas[counter][6] = 0
                        encoded_datas[counter][7] = 0
                    elif np.equal(int(data[counter][i]), 2):
                        print('a1_value=2', int(data[counter][i]))
                        encoded_datas[counter][0] = 0
                        encoded_datas[counter][6] = 1
                        encoded_datas[counter][7] = 0
                    elif np.equal(int(data[counter][i]), 3):
                        print('a1_value=3', int(data[counter][i]))
                        encoded_datas[counter][0] = 0
                        encoded_datas[counter][6] = 0
                        encoded_datas[counter][7] = 1
                        print(encoded_datas[counter][7], 'should be 1')
                # attributo a2
                elif i is 2:
                    if np.equal(int(data[counter][i]), 1):
                        print('a2_value=1', int(data[counter][i]))
                        encoded_datas[counter][1] = 1
                        encoded_datas[counter][8] = 0
                        encoded_datas[counter][9] = 0
                    elif np.equal(int(data[counter][i]), 2):
                        encoded_datas[counter][1] = 0
                        encoded_datas[counter][8] = 1
                        encoded_datas[counter][9] = 0
                    elif np.equal(int(data[counter][i]), 3):
                        encoded_datas[counter][1] = 0
                        encoded_datas[counter][8] = 0
                        encoded_datas[counter][9] = 1
                # attributo a3
                elif i is 3:
                    if np.equal(int(data[counter][i]), 1):
                        encoded_datas[counter][2] = 1
                        encoded_datas[counter][10] = 0
                    elif np.equal(int(data[counter][i]), 2):
                        encoded_datas[counter][2] = 0
                        encoded_datas[counter][10] = 1
                # attributo a4
                elif i is 4:
                    if np.equal(int(data[counter][i]), 1):
                        encoded_datas[counter][3] = 1
                        encoded_datas[counter][11] = 0
                        encoded_datas[counter][12] = 0
                    elif np.equal(int(data[counter][i]), 2):
                        encoded_datas[counter][3] = 0
                        encoded_datas[counter][11] = 1
                        encoded_datas[counter][12] = 0
                    elif np.equal(int(data[counter][i]), 3):
                        encoded_datas[counter][3] = 0
                        encoded_datas[counter][11] = 0
                        encoded_datas[counter][12] = 1
                # attributo a5
                elif i is 5:
                    if np.equal(int(data[counter][i]), 1):
                        encoded_datas[counter][4] = 1
                        encoded_datas[counter][13] = 0
                        encoded_datas[counter][14] = 0
                        encoded_datas[counter][15] = 0
                    elif np.equal(int(data[counter][i]), 2):
                        encoded_datas[counter][4] = 0
                        encoded_datas[counter][13] = 1
                        encoded_datas[counter][14] = 0
                        encoded_datas[counter][15] = 0
                    elif np.equal(int(data[counter][i]), 3):
                        encoded_datas[counter][4] = 0
                        encoded_datas[counter][13] = 0
                        encoded_datas[counter][14] = 1
                        encoded_datas[counter][15] = 0
                    elif np.equal(int(data[counter][i]), 4):
                        encoded_datas[counter][4] = 0
                        encoded_datas[counter][13] = 0
                        encoded_datas[counter][14] = 0
                        encoded_datas[counter][15] = 1
                # attributo a6
                elif i is 6:
                    if np.equal(int(data[counter][i]), 1):
                        encoded_datas[counter][5] = 1
                        encoded_datas[counter][16] = 0
                    elif np.equal(int(data[counter][i]), 2):
                        encoded_datas[counter][5] = 0
                        encoded_datas[counter][16] = 1

        np.savetxt(filename.replace('.train', '_encoded.train'), encoded_datas.astype(np.int64), delimiter=',', fmt='%2.1d')
        return target_x, encoded_datas