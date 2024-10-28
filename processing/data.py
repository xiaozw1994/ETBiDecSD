import processing.config as cfg
import numpy as np
import os 
import matplotlib.pyplot as plt
from keras.utils import  np_utils
import tensorflow as tf
import random
#
#
# reading data from the txt files
#
def readucr(filename):
    data = np.loadtxt(filename+".tsv", delimiter = '\t')
    Y = data[:,0]
    X = data[:,1:]
    if np.isnan(X).any():
        X[np.isnan(X)] = np.nanmean(X)
    return X, Y
###
#    
#  To normalize the trained lebeled data
def NormalizationClassification(Y,num_classes):
    Y = np.array(Y)
    return (Y-Y.mean()) / (Y.max()-Y.mean()) *(num_classes-1)
#
#
#
def NormalizationFeatures(X):
    X = np.array(X)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    value = (X-mean) / std
    return value
#
#  This file is used to get the size of class in the training dataset
#
def GetNumClasses(y):
    y = np.array(y)
    num_classes = len(np.unique(y))
    return num_classes
#### Noising
#   Using the Gussian functi
#
#   To OneHot
#
def OneHot(y,num_classes):
    y = np.array(y)
    y = np_utils.to_categorical(y,num_classes)
    return y
#
#
#Show The index of picure
#
def Show(train_x,aug_x,index,length):
    x = [i for i in range(1,length+1)]
    fig = plt.figure()
    aix = fig.subplots(nrows=2,ncols=1)
    aix[0].plot(x,train_x[index])
    aix[1].plot(x,aug_x[index])
    plt.show()
#
#
# Agumentation 
#
