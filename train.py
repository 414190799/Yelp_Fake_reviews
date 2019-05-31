# coding: utf-8
# !/usr/bin/python3
"""
Authors: Jiajun Bao, Meng Li, Jane Liu
Classes:
    Train: Accepts a list of most frequent topics, trains on the data.
    Feature: Generates features, including length of reviews, calculate percentages of topics in
        fake and real datasets, and finds 10 topic features that have a larger percentage difference
        between real and fake reviews.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

class Train:
    def __init__(self, file, unigramtopics):
        self._train = file
        self._itopic= unigramtopics
        self._len = []
        self._topicf=[]
        self._cols_to_keep=[]

    def Training(self):
        #read file and data cleaning
        train=self._train.dropna()
        train=train[train['content']!="nan"]
        
        #add the feature to the dataframe
        features=Feature(train,self._itopic)
        self._len,self._topicf=features.featuregenerate()
        
        train["length_of_review"]=self._len
        
        for topic in self._topicf:
            tlist=[]
            for c in train["content"]:
                count=0
                p=c.split()
                for word in p:
                    if word==topic:
                        count+=1
                tlist.append(count)
            train[topic]=tlist
            
        # change rating to dummy values
        dummy_ranks = pd.get_dummies(train['rating'], prefix='rating')
        
        #data used to do the regression:
        self._cols_to_keep=["label","length_of_review"]
        
        for elem in self._topicf:
            self._cols_to_keep.append(elem)
        
        data = train[self._cols_to_keep].join(dummy_ranks.ix[:, 'rating_2.0':])
        
        #change the label of fake data from -1 to 0
        data.loc[data.label==-1,'label'] =0
        
        #train columns
        train_cols = data.columns[1:]
        
        #standarized train data
        data[train_cols]=data[train_cols].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        
        #add intercept
        data['intercept'] = 1.0
        
        train_cols1 = data.columns[1:]

        #train the data
        logit = sm.Logit(data['label'].astype(float), data[train_cols1].astype(float))
        result = logit.fit()
        
        print(result.summary())
        return data, self._topicf, result


class Feature:
    
    def __init__(self, train_dataset, topics):
        self._train=train_dataset
        self._topics=topics
        self._len = []
        self._topicf=[]   

    def featuregenerate(self):
        #convert column content to list
        content=[]
        content=self._train.content.tolist()
        #generate feature "length of review"
        listlen=[]
        for elem in content:
            listlen.append(len(str(elem)))
        
        #generate topic features:
        
        #split dataset to fake and non-fake
        nfdata=self._train.loc[self._train["label"]==1]
        fdata=self._train.loc[self._train["label"]==-1]
        
        #data cleaning
        fdata=fdata.dropna()
        nfdata=nfdata.dropna()
        nfdata=nfdata[nfdata['content']!="nan"]
        fdata=fdata[fdata['content']!="nan"]
        
        #convert content to list
        contentn=[]
        contentf=[]
        contentn=nfdata.content.tolist()
        contentf=fdata["content"].tolist()
        
        #calculate percentages of topics respectively in fake dataset and non-fake dataset 
        
        topicdict={}
        ftopicdict={}
        
        for topic in self._topics:
            count=0
            for elem in range(len(contentn)):
                if topic in contentn[elem]:
                     count+=1
            topicdict[topic]=count/len(contentn)
            
        for topic in self._topics:
            count=0
            for elem in range(len(contentf)):
                if topic in contentf[elem]:
                    count+=1
            ftopicdict[topic]=count/len(contentf)
            
        #calculate difference value of each topic between its percentages in fake and real datasets
        topicdiff={}
        for key in topicdict:
            topicdiff[key]=topicdict[key]-ftopicdict[key]
        for key in topicdiff:
            if topicdiff[key]<0:
                topicdiff[key]=-topicdiff[key]
                
        #generate 10 topic features which has the bigger difference value of percentage between fake and non-fake:        
        listk=[]
        listk=sorted(topicdiff.items(), key=lambda d: d[1],reverse=True)
        listtopic=[]
        for elem in listk[:10]:
            listtopic.append(elem[0])
        
        return listlen, listtopic
        
        
