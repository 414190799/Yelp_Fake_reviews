# coding: utf-8
# !/usr/bin/python3
"""
Authors: Jiajun Bao, Meng Li, Jane Liu
Classes:
    NN: Accepts the preprocessed data and trains a Multilayer Perceptron (MLP) classifier. Reports the accuracy of the MLP model.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class NN:
    #import the train dataset and validate dataset after feature generating
    def __init__(self,train,val):
        self._train = train
        self._val= val
    
    def train(self):

        dataMat = np.array(self._train)
        
        #assign x and y
        X=dataMat[:,1:-1]
        y = dataMat[:,0]
        
        #standarize and transform data
        scaler = StandardScaler() 
        scaler.fit(X) 
        X = scaler.transform(X)
        
        #train the model. hidden_layer_sizes=(50,20) which will get best results
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50,20), random_state=1) 
        clf.fit(X, y)
        
        #test the model
        valMat = np.array(self._val)
        X2=valMat[:,1:-3]
        y2 = valMat[:,0]
        scaler = StandardScaler() 
        scaler.fit(X2) 
        X2 = scaler.transform(X2)
        
        #return the accuracy result
        print("Neural Network accuracy score: ", clf.score(X2,y2))
        
