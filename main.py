# coding: utf-8
# !/usr/bin/python3
"""
Authors: Jiajun Bao, Meng Li, Jane Liu
Classes:
    Main: Explore the dataset, merge the data, perform preprocessing, pass training data to the models.

"""

import numpy as np
import pandas as pd
from dataset import *
from train import *
from validate import *
from preprocessor import *
from neuralnetwork import *
from detecting_sys_window import *

def main():
    
    # perform data exploration then split the data into train, validate and test sets
    Prep = Dataset('data/metadata.txt','data/reviewContent.txt')
    Prep.data_explore()
    t,v,test = Prep.bdsproject_merge()
    
    # output the most frequent unigrams
    prep_data = Preprocessor()
    unigramtopics = prep_data.preprocess()
    ngram_print(unigramtopics)    # print the most frequent unigrams to ngrams.txt
    
    # train the logistic regression model using ngrams only
    Tra = Train(t, unigramtopics)
    data,topicf,result = Tra.Training()
    
    # test and print the result of confusion matrix
    vali = Validate(v, topicf, result)
    valdata = vali.valid()
    
    # use MLPClassifier of NN to train the model and get the accuracy
    NNt = NN(data, valdata)
    NNt.train()

    #run the detecting window
    # frame = MainWindow(v, topicf, result)
 
    
main()
