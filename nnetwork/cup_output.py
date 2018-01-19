# -*- coding: utf-8 -*-

# ==============================================================================
# Codice per creare il file txt della cup
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================

"""
FORMATO:

# name of the candidates
# nickname of the team (max 8-10 char) for the results page on the web
# name of the  data set (ML-CUP17) - ML 2017 CUP v1
# date

315 righe
ID, output_x, output_y
"""

import numpy as np


"""
test_matrix = Test set
out_vect = output della reg_term
filename = nome file o path
header = header da inserire all'inizio del file
"""
def save_cup_out(test_matrix, out_vect, filename, header):
    ids = test_matrix[:, 0]
    ids = np.reshape(ids, [ids.shape[0], -1])
    cup_out = np.concatenate((ids, out_vect), axis=1)
    np.savetxt(filename, cup_out, header=header, comments='', fmt="%.0d,%.6f,%.6f")


"""
Carica i dati della ML cup
numpy elimina automaticamente l'header
ritorna il training set senza il campo id
"""
def preprocess_cup_train(filename):
    raw_datas = open(filename, 'r')
    tr_set = np.loadtxt(raw_datas, delimiter=',')
    tr_set = np.delete(tr_set, 0, axis=1)  # rimuoviamo colonna degli id
    return tr_set


"""
In questa funzione vanno inserite le informazioni
"""
def __main__():
    # test save_cup_out function
    header = (
    "# Michele Resta, Lavinia Salicchi\n"
    "# nickname of the team (max 8-10 char) for the results page on the web\n"
    "# name of the  data set (ML-CUP17) - ML 2017 CUP v1\n"
    "# date\n"
    )
    filename = "team-name_ML-CUP17-TS.csv"
    tr = preprocess_cup_train('/Users/mick/Dati/Università/Pisa/Machine_learning/Prj_info/ML-CUP17-TR.csv')
    raw_data = open('/Users/mick/Dati/Università/Pisa/Machine_learning/Prj_info/ML-CUP17-TS.csv', 'r')
    test_matrix = np.loadtxt(raw_data, delimiter=',')
    out_vect = test_matrix[:, 1:3]
    save_cup_out(test_matrix, out_vect, filename, header)

    # end test save_cup_out function


__main__()
