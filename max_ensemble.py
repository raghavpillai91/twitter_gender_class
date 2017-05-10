# -*- coding: utf-8 -*-
__author__ = "Raghav Pillai"

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
import argparse
from sklearn import svm
# import xgboost
from sklearn.neural_network import MLPClassifier
from gensim import models
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from bs4 import BeautifulSoup
import re
from data_cleaning import CleanData
from nltk.corpus import stopwords
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

Word2vecModel = models.Word2Vec.load("gender_Word2vec_Vector")

class max_ensemble(object):

    def create_datavecs(self,sentences):
        num_features=300
        DataVecs = self.getAvgFeatureVecs(sentences, Word2vecModel, num_features)
        return DataVecs

    def create_engine(self,sentences,train_class):
        num_features=300
        DataVecs = self.getAvgFeatureVecs(sentences, Word2vecModel, num_features)
        clfs = VotingClassifier(estimators=[('rfgini',RandomForestClassifier(n_estimators=220,max_depth=10, n_jobs=-1, criterion='gini')),
                                            ('rfentropy',RandomForestClassifier(n_estimators=120,max_depth=10, n_jobs=-1, criterion='entropy')),
                                            ('etgini',ExtraTreesClassifier(n_estimators=530,max_depth=10, n_jobs=-1, criterion='gini')),
                                            ('etgini', ExtraTreesClassifier(n_estimators=200, max_depth=10, n_jobs=-1,criterion='gini')),
                                            ('svmlinear',svm.SVC(kernel='linear',C=1.0,probability=True))
                                            ], voting='soft')
        clfs=clfs.fit(DataVecs,train_class)
        joblib.dump(clfs, 'max_ensembled_gender_300.pk')

    @staticmethod
    def getAvgFeatureVecs(reviews,word_vector_news, num_features):
        count = 0
        feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")
        for sentence in reviews:
            feature_vecs[count] = max_ensemble.makeFeatureVec(sentence,word_vector_news,
                                                   num_features=300)
            count = count + 1
        return feature_vecs


    @staticmethod
    def makeFeatureVec(words, model,num_features):
        feature_vec = np.zeros((num_features,), dtype="float32")
        # print sentence
        ehst_vec_set = set(model.wv.index2word)
        count = 0
        for word in words:
            if word in ehst_vec_set:
                count = count + 1
                vector_w2vec = model[word]
                feature_vec = np.add(feature_vec, vector_w2vec)

        if count == 0:
            feature_vec = np.ones((num_features,), dtype="float32")
        else:
            feature_vec = np.divide(feature_vec, count)
        return feature_vec

    @staticmethod
    def create_ngrams(text):
        vectorizer = CountVectorizer(ngram_range=(1, 3))
        analyzer = vectorizer.build_analyzer()
        return analyzer(text)

    def predict_input(self,testDataVec):
        clf = joblib.load('max_ensembled_gender_300.pk')
        result=clf.predict(testDataVec)
        print result
    def makeTestData(self,sentences):
        num_features = 300
        DataVecs = max_ensemble.getAvgFeatureVecs(sentences, Word2vecModel, num_features)
        return DataVecs

if __name__ == '__main__':
    # create object for the parser
    parser = argparse.ArgumentParser()

    # create option to read the query string
    parser.add_argument('-q', '--q', help='Train the model', nargs='+')
    parser.add_argument('-t', '--t', help='test string input', nargs='+')
    # parsing arguments from commandline
    arguments = parser.parse_args()

    clean_obj = CleanData()
    max_obj = max_ensemble()

    if arguments.q:
        training_file = 'data_gender.csv'
        def read_data(training_file):
            train = pd.read_csv(training_file,sep=',')
            return train
        print 'read training file'
        train = read_data(training_file=training_file)
        clean_sentences = []
        for sentence in train['title']:
            sen = ' '.join(clean_obj.clean_article(sentence))
            clean_sentences.append(max_obj.create_ngrams(sen))

        train_class = train['Gender']
        max_obj.create_engine(clean_sentences,train_class)
    elif arguments.t:
        test = ' '.join(arguments.t)
        clean_sentences = []
        sen = ' '.join(clean_obj.clean_article(test))
        clean_sentences.append(max_obj.create_ngrams(sen))
        test_datavecs = max_obj.makeTestData(clean_sentences)
        max_obj.predict_input(test_datavecs)