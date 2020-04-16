#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:58:59 2020

@author: vishal
"""

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import tree
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

wpt = nltk.WordPunctTokenizer()
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
stop_words.extend(["et","al"])

data = pd.read_json("dev.json");
strings = data["string"]
label = data["label"]

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)

# Normalize data
norm_corpus = normalize_corpus(strings)

# Create Bag of Words
cv = CountVectorizer(min_df=0., max_df=1.)
cv_matrix = cv.fit_transform(norm_corpus)
cv_matrix = cv_matrix.toarray()
vocab = cv.get_feature_names()
bow_df = pd.DataFrame(cv_matrix, columns=vocab)

# Create Bag of N Grams
bv = CountVectorizer(ngram_range=(2,2))
bv_matrix = bv.fit_transform(norm_corpus)

bv_matrix = bv_matrix.toarray()
vocab = bv.get_feature_names()
bonw_df = pd.DataFrame(bv_matrix, columns=vocab)

# Create TF IDF Vectors
tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
tv_matrix = tv.fit_transform(norm_corpus)
tv_matrix = tv_matrix.toarray()

vocab = tv.get_feature_names()
tv_df = pd.DataFrame(np.round(tv_matrix, 2), columns=vocab)

# Count Cosine Similarity
similarity_matrix = cosine_similarity(tv_matrix)
cosine_similarity_df = pd.DataFrame(similarity_matrix)
    
dt = tree.DecisionTreeClassifier()
dt = dt.fit(cosine_similarity_df, label)

Z = linkage(similarity_matrix, 'ward')
#dt.predict(["The dosages of L-NMMA and indomethacin infused were selected on the basis of previous experiments that found a reduction in muscle blood flow during exercise (28)."])

plt.figure(figsize=(8, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
dendrogram(Z)
plt.axhline(y=3.2, c='k', ls='--', lw=0.5)

max_dist = 3.2

cluster_labels = fcluster(Z, max_dist, criterion='distance')
cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
clusters = pd.concat([pd.DataFrame(norm_corpus), pd.DataFrame(label), cluster_labels], axis=1)
