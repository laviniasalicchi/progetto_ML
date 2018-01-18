# -*- coding: utf-8 -*-

# ==============================================================================
# QUesta classe utilizza un ThreadPoolExecutor per la ricerca degli
# iperparametri nella rete neurale
#
# © 2017 Mick Hardins & Lavinia Salicchi
# ==============================================================================

from cross_validation import *
import concurrent.futures
import random
import time




def __main__():

    # creo il ThreadPoolExecutor: maxpool = numero processori * 5
    executor = concurrent.futures.ThreadPoolExecutor()
    for i in range(0, 10):
        rand = random.randint(0, 10)
        name = "Thread" + str(i)
        executor.submit(countTest, rand, name)
        #res = future.result()
    print("submission ended")
    executor.shutdown(wait=True)
    input()





def countTest(start, name):
    #print("ass")
    for i in reversed(range(0, start)):
        print("Hello from thread ", name, str(i), "second")
        time.sleep(1)


__main__()
