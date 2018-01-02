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

filename = 'monks-1.train'
fields = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']
#data = np.loadtxt(raw_data, delimiter=",")

target_x = np.empty(data.shape[0], 1)
encoded_datas = np.empty(data.shape[0], 17)

with open(filename, newline='') as csvfile:
    reader = csv.DictReader(csvfile, fields)
    row_count = sum(1 for row in reader)
    i = 0
    for row in reader:
        if row['a1'] is 1:
            encoded_datas[i][0] = 1
            encoded_datas[i][6] = 0
            encoded_datas[i][7] = 0
        elif row['a1'] is 2:
            encoded_datas[i][0] = 0
            encoded_datas[i][6] = 1
            encoded_datas[i][7] = 0
        elif row['a1'] is 3:
            encoded_datas[i][0] = 0
            encoded_datas[i][6] = 0
            encoded_datas[i][7] = 1
        if row['a2'] is 1:
            encoded_datas[i][1] = 1
            encoded_datas[i][8] = 0
            encoded_datas[i][9] = 0
        elif row['a2'] is 2:
            encoded_datas[i][1] = 0
            encoded_datas[i][8] = 1
            encoded_datas[i][9] = 0
        elif row['a2'] is 3:
            encoded_datas[i][1] = 0
            encoded_datas[i][8] = 0
            encoded_datas[i][9] = 1
        if row['a3'] is 1:
            encoded_datas[i][2] = 1
            encoded_datas[i][10] = 0
        elif row['a3'] is 2:
            encoded_datas[i][2] = 0
            encoded_datas[i][10] = 1
        if row['a4'] is 1:
            encoded_datas[i][3] = 1
            encoded_datas[i][11] = 0
            encoded_datas[i][12] = 0
        elif row['a4'] is 2:
            encoded_datas[i][3] = 0
            encoded_datas[i][11] = 1
            encoded_datas[i][12] = 0
        elif row['a4'] is 3:
            encoded_datas[i][3] = 0
            encoded_datas[i][11] = 0
            encoded_datas[i][12] = 1
        if row['a5'] is 1:
            encoded_datas[i][4] = 1
            encoded_datas[i][13] = 0
            encoded_datas[i][14] = 0
            encoded_datas[i][15] = 0
        elif row['a5'] is 2:
            encoded_datas[i][4] = 0
            encoded_datas[i][13] = 1
            encoded_datas[i][14] = 0
            encoded_datas[i][15] = 0
        elif row['a5'] is 3:
            encoded_datas[i][4] = 0
            encoded_datas[i][13] = 0
            encoded_datas[i][14] = 1
            encoded_datas[i][15] = 0
        elif row['a5'] is 4:
            encoded_datas[i][4] = 0
            encoded_datas[i][13] = 0
            encoded_datas[i][14] = 0
            encoded_datas[i][15] = 1
        if row['a6'] is 1:
            encoded_datas[i][5] = 1
            encoded_datas[i][16] = 0
        elif row['a6'] is 2:
            encoded_datas[i][5] = 0
            encoded_datas[i][16] = 1
        target_x[i][0] = row['class']
        i = i +1

print(target_x)
print (encoded_datas)
