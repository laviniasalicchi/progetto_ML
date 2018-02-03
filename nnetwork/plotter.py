# -*- coding: utf-8 -*-

# ==============================================================================
# Classe per plottre grafici
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # mac osx need this backend



class Plotter:

    @staticmethod
    def plot_kfold(kfold_res):
        """

        Plotta i risultati di una kfold in un unico grafico
        """

        tr_folds_err_h = kfold_res['tr_folds_err_h']
        vl_folds_err_h = kfold_res['vl_folds_err_h']
        epochs_count = len(tr_folds_err_h[0])
        k = len(tr_folds_err_h) # fold number
        epochs_l = range(0, epochs_count)
        tr_arr = np.array(tr_folds_err_h)
        tr_mean_err = np.mean(tr_arr, axis=0)
        vl_arr = np.array(vl_folds_err_h)
        vl_mean_err = np.mean(vl_arr, axis=0)
        for i in range(0, k):
            tr_err = tr_folds_err_h[i]
            vl_err = vl_folds_err_h[i]
            plt.plot(epochs_l, tr_err, color='xkcd:sky')
            plt.plot(epochs_l, vl_err, color='xkcd:peach')
        plt.plot(epochs_l, tr_mean_err, color='xkcd:blue', label='avg training err')
        plt.plot(epochs_l, vl_mean_err, color='xkcd:red', label='avg val err')

        plt.xlabel("epochs")
        plt.ylabel("error")
        plt.legend(loc='upper right', frameon=False)
        plt.show()


    @staticmethod
    def plotError(epochs_plot, errors, ts_error, folder):
        plt.plot(epochs_plot, errors, color="blue", label="training error")
        plt.plot(epochs_plot, ts_error, color="red", label="test error", linestyle="-.")
        plt.xlabel("epochs")
        plt.ylabel("error")
        plt.legend(loc='upper right', frameon=False)

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
        plt.legend(loc='lower right', frameon=False)

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
        plt.legend(loc='upper right', frameon=False)

        plt.show()

    @staticmethod
    def plot_accuracy_noTS(epochs_plot, accuracy):
        plt.plot(epochs_plot, accuracy, color="blue", label="accuracy TR")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend(loc='lower right', frameon=False)
        plt.show()
