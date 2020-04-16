#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 22:16:13 2020

@author: vishal
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix

data = pd.read_json("dev.json");
strings = list(data["string"])
label = list(data["label"])

# Extracting features from text files
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(strings)
X_train_counts.shape

# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

# Machine Learning
# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, label)

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(strings, label)

test_data = pd.read_json("test.json");
test_strings = list(test_data["string"])
test_label = list(test_data["label"])

# Performance of NB Classifier
predicted = text_clf.predict(test_strings)
print(np.mean(predicted == test_label))
print("NB Classification Report:")
print(classification_report(test_label,predicted))

# Training Support Vector Machines - SVM and calculating its performance

from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42))])

text_clf_svm = text_clf_svm.fit(strings, label)
predicted_svm = text_clf_svm.predict(test_strings)
print(np.mean(predicted_svm == test_label))
print("SVM Classification Report:")
print(classification_report(test_label,predicted_svm))

