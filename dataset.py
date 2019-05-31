# coding: utf-8
# !/usr/bin/python3
"""
Authors: Jiajun Bao, Meng Li, Jane Liu
Classes:
    Dataset: Explore the data, merge the datasets and split it into a training set, test set, and validation set.
"""

# In[17]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Dataset:

    def __init__(self, txt1,txt2):
        self._txt1 = txt1
        self._txt2 = txt2
        self._train=pd.DataFrame()
        self._validate=pd.DataFrame()
        self._test=pd.DataFrame()

    def data_explore(self):
        xpnames1 = ["userID", "b", "rating", "label", "date"]
        xpnames2 = ["userID", "b", "date", "content"]
        xpdata1 = pd.read_csv(self._txt1, sep="\t", names=xpnames1)
        xpdata2 = pd.read_csv(self._txt2, sep="\t", names=xpnames2)

        # explore metadata.txt
        print("Exploring the metadata.txt file: \n\n(the first 10 entries of metadata.txt)\n", xpdata1.head(10))
        print(xpdata1.describe(include="all"))
        print("\n")

        # explore reviewContent.txt
        print("Exploring the reviewContent.txt file: \n\n(first 10 entries of reviewContent.txt)\n", xpdata2.head(10))
        print(xpdata2.describe(include="all"))
        print("\n")

        print("The distribution of Yelp review ratings:\n", xpdata1["rating"].value_counts())
        print("\nThe number of reviewers (spammers are indicated by '-1' value):\n", xpdata1["label"].value_counts())
        print("\n")

        # Prince cluster does not generate matplotlib whisker plots and histograms
        # xpdata1.boxplot()
        # xpdata1.hist()

    def bdsproject_merge(self):
        #merge dataset by userID
        data = pd.read_csv(self._txt1, sep="\t", header=None)
        data.columns = ["userID","b", "rating", "label", "date"]
        data=data.drop(['b', 'date'],axis=1)
        data2 = pd.read_csv(self._txt2, sep="\t", header=None)
        data2.columns = ["userID","b", "date", "content"]
        result=data.set_index('userID').join(data2.set_index('userID'))
        
        #rename column
        result.columns = [ "rating", "label", "prob_ID","date","content"]
        
        #make the dataset balanced
        fakedata1=result.loc[result["label"]==-1]
        nfakedata1=result.loc[result["label"]==1]
        nfakerdata2=nfakedata1.sample(n=105000)
        result2=pd.concat([fakedata1,nfakerdata2],ignore_index=True)
        
        #split dataset to train, validate and test and save it as csv.
        self._train, self._validate, self._test = np.split(result2.sample(frac=1), [int(.6*len(result2)), int(.8*len(result2))])
        
        self._train.to_csv("data/train.csv")
        self._validate.to_csv("data/validate.csv")
        self._test.to_csv("data/test.csv")
        
        return self._train, self._validate, self._test

        


