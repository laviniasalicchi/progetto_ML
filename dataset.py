import csv
import numpy as np
#import pandas

'''numpy'''
filename = 'ML-CUP17-TR.csv'
raw_data = open(filename, 'r')

data = np.loadtxt(raw_data, delimiter=",")

x = np.empty([1016,10])
target_x = np.empty([1016,1])
target_y = np.empty([1016,1])

print(type(x))

for i in range(0, len(data[:, 0])):
    print("** ",i," **")
    k = 0
    for j in range(1,11):
        print(j," - ", data[i][j])
        x[i][k] = data[i][j]
        k = k+1
    target_x[i][0] = data[i][11]
    target_y[i][0] = data[i][12]

print("___________X__________")
print(x)
print("___________TARGET_X__________")
print(target_x)
print("___________TARGET_Y__________")
print(target_y)



'''pandas
filename = 'ML-CUP17-TR.csv'
names = ['id','input1','input2','input3','input4','input5','input6','input7','input8','input9','input10','target_x','target_y']
data = pandas.read_csv(filename, names=names)
print(type(data))
'''
