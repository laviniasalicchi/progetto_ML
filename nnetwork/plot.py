import matplotlib.pyplot as plt
import numpy as np

# array con il numero di ogni epoca
epochs = 5
epoch = []
for ep in range(epochs):
    epoch.append(ep+1)


#array con valori degli errori per ogni epoca
err_training = [2, 3, 4.5, 1, 0.5]
err_test = [2, 3.5, 5, 2, 1]

# plot con x =  epoche ; y = val errore
plt.plot(epoch, err_training, color = "blue", label = "training error")
plt.plot(epoch, err_test, color = "red", label = "test error", linestyle = "-.")

plt.xlabel("epochs")
plt.ylabel("error")

plt.legend(loc='upper left', frameon=False)

plt.show()
