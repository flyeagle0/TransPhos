import functools
import itertools
import os
import random
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Reshape
from tensorflow.keras.layers import Conv1D,Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2


import copy

def predict(train_file_name,sites,dataset=None,
                         hierarchy=None, kinase=None):
    


    win1 = 51
    win2 = 33
    
    from dataprocess_predict import getMatrixInput
    [X_test1,y_test,ids,position,label] = getMatrixInput(train_file_name, sites, win1)
    [X_test2,_,_,_,_] = getMatrixInput(train_file_name, sites, win2)
    

    from transnet import transphos
    model = transphos(X_test1, X_test2, y_test)
    
    
    
    model_weight = '/home/zhiyuan/TransPhos/model/PELM_' + sites + ".h5"
    model.load_weights(model_weight)
    

    predictions_t = model.predict([X_test1, X_test2])
    results_ST = np.column_stack((predictions_t[:, 1],label))

    result = pd.DataFrame(results_ST)

    outputfile = "result/"
    result.to_csv(outputfile + dataset +"_"+sites+"_val.txt", index=False, header=None, sep='\t',
                  quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':
    train_file_name = '/home/zhiyuan/TransPhos/dataset/'
    site = 'Y'

    dataset = "PELM_"
    predict(train_file_name+dataset+site+"_val.csv", site,dataset)



