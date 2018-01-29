# -*- coding: utf-8 -*-

# ==============================================================================
# Classe per plottre grafici
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================

import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # mac osx need this backend



class Plotter:

    @staticmethod
    def plotError(epochs_plot, errors, ts_error, folder):
        plt.plot(epochs_plot, errors, color="blue", label="training error")
        plt.plot(epochs_plot, ts_error, color="red", label="test error", linestyle="-.")
        plt.xlabel("epochs")
        plt.ylabel("error")
        plt.legend(loc='upper left', frameon=False)

        path = folder+"plots/"
        if not os.path.exists(path):
            os.makedirs(path)
        file = path+"err.png"
        plt.savefig(file)
        plt.show()

    @staticmethod
    def plot_accuracy(epochs_plot, accuracy, ts_accuracy, folder):
        plt.plot(epochs_plot, accuracy, color="blue", label="accuracy TR")
        plt.plot(epochs_plot, ts_accuracy, color="red", label="accuracy TS", linestyle="-.")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend(loc='upper left', frameon=False)

        path = folder + "plots/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = path+"acc.png"
        plt.savefig(file)
        plt.show()

    @staticmethod
    def plotError_noTS(epochs_plot, errors):
        plt.plot(epochs_plot, errors, color="blue", label="training error")
        plt.xlabel("epochs")
        plt.ylabel("error")
        plt.legend(loc='upper left', frameon=False)

        plt.show()

    @staticmethod
    def plot_accuracy_noTS(epochs_plot, accuracy):
        plt.plot(epochs_plot, accuracy, color="blue", label="accuracy TR")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend(loc='upper left', frameon=False)
        plt.show()