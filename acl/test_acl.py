import pickle
import numpy as np
from sklearn.metrics import classification_report
from random import shuffle
import pandas as pd
from sklearn.model_selection import KFold


def main():
    citations = []
    labels = []
    cite_context = []
    with open('citations.pkl', 'rb') as f:
        citations = pickle.load(f)
    with open('labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open('cite_context.pkl', 'rb') as f:
        cite_context = pickle.load(f)
    # print(citations)
    # print(labels)
    # print(cite_context)

    final = list(zip(labels, cite_context))
    shuffle(final)
    labels, cite_context = zip(*final)

    print("total length: ", len(citations))
    split_percent = 0.7
    split_index = int(len(citations) * split_percent)
    print("train split percentage: ", split_percent)
    train_silce = slice(0, split_index)
    test_silce = slice(split_index, len(citations) - 1)
    # kf = KFold(n_splits=10)
    # kf.get_n_splits(cite_context)

    strings = cite_context[train_silce]
    label = labels[train_silce]

    test_strings = cite_context[test_silce]
    test_label = labels[test_silce]

    ###start####
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

    ###classification-start###
    # Performance of NB Classifier
    predicted = text_clf.predict(test_strings)
    print(np.mean(predicted == test_label))
    print("NB Classification Report:")
    print(classification_report(test_label, predicted))

    # Training Support Vector Machines - SVM and calculating its performance

    from sklearn.linear_model import SGDClassifier
    text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                             ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))])

    text_clf_svm = text_clf_svm.fit(strings, label)
    predicted_svm = text_clf_svm.predict(test_strings)
    print(np.mean(predicted_svm == test_label))
    print("SVM Classification Report:")
    print(classification_report(test_label, predicted_svm))

    # Grid Search
    from sklearn.model_selection import GridSearchCV
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(strings, label)
    predicted_gs = gs_clf.predict(test_strings)
    print("Grid Search Performance:")
    print(classification_report(test_label, predicted_gs))
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)

    # Similarly doing grid search for SVM
    from sklearn.model_selection import GridSearchCV
    parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),
                      'clf-svm__alpha': (1e-2, 1e-3)}

    gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
    gs_clf_svm = gs_clf_svm.fit(strings, label)
    predicted_gs_svm = gs_clf_svm.predict(test_strings)
    print("Grid Search Performance for SVM:")
    print(classification_report(test_label, predicted_gs_svm))
    print(gs_clf_svm.best_score_)
    print(gs_clf_svm.best_params_)

    # NLTK
    # Removing stop words
    from sklearn.pipeline import Pipeline
    text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())])

    # Stemming Code

    import nltk
    nltk.download('stopwords')

    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english", ignore_stopwords=True)

    class StemmedCountVectorizer(CountVectorizer):
        def build_analyzer(self):
            analyzer = super(StemmedCountVectorizer, self).build_analyzer()
            return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

    stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

    text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                                 ('mnb', MultinomialNB(fit_prior=False))])

    text_mnb_stemmed = text_mnb_stemmed.fit(strings, label)

    predicted_mnb_stemmed = text_mnb_stemmed.predict(test_strings)
    print("NLTK Performance:")
    print(classification_report(test_label, predicted_mnb_stemmed))


if __name__ == '__main__':
    main()
