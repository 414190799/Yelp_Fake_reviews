#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import string
import unittest
from collections import Counter
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

np.random.seed(2018)

import nltk

from gensim import corpora

Lda = gensim.models.ldamodel.LdaModel

num_topic = 5  # number of topics selected.
k = 5  # k value for Knn


class Tokenizer:
    def __init__(self, text):
        self._text = text
        self._tokenized_text = []

    def tokenize(self):
        # Remove punctuation, empty elements, numbers, and dates
        pattern = re.compile('[0-9]+')
        self._tokenized_text = [''.join(c for c in s if c not in string.punctuation) for s in self._text]
        self._tokenized_text[:] = [word for word in self._tokenized_text if not pattern.match(word) and word != '']

        return self._tokenized_text


class RemoveStopWords:
    def __init__(self, text):
        self._text = text
        self._stopwords = []

    def removestopwords(self):
        with open('data/stopwords.txt', 'r') as g:
            self._stopwords = g.read().splitlines()
        for word in self._stopwords:
            self._text = [value for value in self._text if value.lower() != word]

        return self._text


class dataset:
    def __init__(self, txt1, txt2):
        self._txt1 = txt1
        self._txt2 = txt2
        self._train = pd.DataFrame()
        self._validate = pd.DataFrame()
        self._test = pd.DataFrame()

    def bdsproject_merge(self):
        # merge dataset by userID
        data = pd.read_csv(self._txt1, sep="\t", header=None)
        data.columns = ["userID", "b", "rating", "lable", "date"]
        data = data.drop(['b', "rating", 'date'], axis=1)
        data2 = pd.read_csv(self._txt2, sep="\t", header=None)
        data2.columns = ["userID", "b", "date", "content"]
        data2 = data2.drop(['b', 'date'], axis=1)
        result = data.set_index('userID').join(data2.set_index('userID'))

        # rename column
        result.columns = ["label", "content"]

        # make the dataset balanced
        fakedata1 = result.loc[result["label"] == -1]
        fakedata2 = fakedata1.sample(n=10000)
        nfakedata1 = result.loc[result["label"] == 1]
        nfakedata2 = nfakedata1.sample(n=10000)
        nfakedata3 = nfakedata1.sample(n=105000)
        result2 = pd.concat([fakedata1, nfakedata3], ignore_index=True)

        # split dataset to train, validate and test and save it as csv.

        train, validate, test = np.split(result2.sample(frac=1), [int(.6 * len(result2)), int(.8 * len(result2))])
        test = test.sample(n=10000)
        final = pd.concat([fakedata2, nfakedata2, test])
        print(final)

        return final


def textprocess(data):
    data = data.content.values.tolist()
    lemmatizer = WordNetLemmatizer()
    wordCollect = []
    for word in data:
        word = str(word)
        wordCollect.append(word.split())
    collect = []
    for data in wordCollect:
        preprocessedlist = []
        temptext = Tokenizer(data)
        cleantext = temptext.tokenize()
        temptext = RemoveStopWords(cleantext)
        cleantext = temptext.removestopwords()
        lemma_text = []

        for word in list(cleantext):
            new_word = lemmatizer.lemmatize(word)
            lemma_text.append(new_word)

        for word in lemma_text:
            preprocessedlist.append(word)
        collect.append(preprocessedlist)
    return collect


def lda(collect):
    # dic for terms
    dictionary = corpora.Dictionary(collect)

    # DT matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in collect]
    return dictionary, doc_term_matrix


Prep = dataset('data/metadata.txt', 'data/reviewContent.txt')

f = Prep.bdsproject_merge()
fc = textprocess(f)

mark = f.label.values.tolist()
dictionary_f, corpus_f = lda(fc)
lda_f = Lda(corpus_f, num_topics=num_topic, id2word=dictionary_f, passes=50)
topics = lda_f.show_topics()
for topic in topics:
    print(topic)

all_topics = lda_f.get_document_topics(corpus_f, per_word_topics=True)
i = 0
d_t = []
for doc_topics, word_topics, phi_values in all_topics:
    i += 1
    d_t.append(doc_topics)
    print('Review ' + str(i) + ' topics:', doc_topics)

print(lda_f)

vec = []
for term in d_t:
    temp = []
    for i in range(0, num_topic):
        temp.append(0)
    for topic in term:
        temp[int(topic[0])] = topic[1]
    vec.append(temp)


def Knn(a, train, k):
    i = 0
    mi = []
    index = []
    for x in range(0, k):
        mi.append(100)
        index.append(0)
    for l in train:
        dis = dist(a, l)
        if dis < mi[-1]:
            mi[k - 1] = dis
            index[k - 1] = i
            j = k - 1
            while mi[j] < mi[j - 1] and j >= 1:
                temp = mi[j]
                mi[j] = mi[j - 1]
                mi[j - 1] = temp
                temp = index[j]
                index[j] = index[j - 1]
                index[j - 1] = temp
                j -= 1

        i += 1
    return index


def dist(a, b):
    dis = 0
    for i in range(0, num_topic):
        dis += ((a[i] - b[i]) ** 2) ** 0.5
    return dis


# In[2]:


train = vec[:20000]
test = vec[20000:]
i = 20000
tf = 0
tt = 0
ft = 0
ff = 0
for term in test:
    best = Knn(term, train, k)
    for index in best:
        count_t = 0
        count_f = 0
        if mark[index] == -1:
            count_f += 1
        else:
            count_t += 1
    if count_f > count_t:
        r = -1
    else:
        r = 1
    if r == mark[i]:
        print(True)
        if r == 1:
            tt += 1
        else:
            ff += 1
    else:
        print(False, best[0], best[1], best[2], r)
        if r == 1:
            tf += 1
        else:
            ft += 1
    i += 1

# # In[3]:
#
#
# tt
# tf
# ff
# ft
#
# # In[4]:
#
#
# tt
#
# # In[5]:
#
#
# tf
#
# # In[6]:
#
#
# ff
#
# # In[7]:
#
#
# ft
#
# # In[8]:
#
#
print((tt + ff) / (ft + tf + tt + ff))
#
#
# # In[ ]: