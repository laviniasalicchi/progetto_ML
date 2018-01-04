# -*- coding: utf-8 -*-

# ==============================================================================
# E' una hidden layer
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================

import numpy as np

class ML_CUP_Dataset:

    #filename = 'ML-CUP17-TR.csv'
    def load_ML_dataset(filename):
        raw_data = open(filename, 'r')

        data = np.loadtxt(raw_data, delimiter=",")

        x = np.empty([data.shape[0], data.shape[1] - 3])
        target_x = np.empty([data.shape[0], 1])
        target_y = np.empty([data.shape[0], 1])
        target_values = np.empty([data.shape[0],    2])   # // target values = (pattern, target_x/y)

        for i in range(0, len(data[:, 0])):
            k = 0
            for j in range(1,11):
                x[i][k] = data[i][j]
                k = k+1
            target_x[i][0] = data[i][11]
            target_y[i][0] = data[i][12]
            target_values[i][0] = data[i][11]
            target_values[i][1] = data[i][12]

        target_values = np.transpose(target_values)
        x = np.transpose(x)

        return x, target_values