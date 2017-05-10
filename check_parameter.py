# -*- coding: utf-8 -*-
__author__ = "Raghav Pillai"

import logging
logging.getLogger().setLevel(logging.INFO)

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from data_cleaning import CleanData
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV


clean_obj = CleanData()
training_file = 'data_gender.csv'

def read_data(training_file):
    train = pd.read_csv(training_file,sep=',')
    return train

def clean_reviews(sentence):
    clean_sentence = clean_obj.clean_article(sentence)
    return clean_sentence

def make_feature_vec(sentence,ehst_model,num_features):
    feature_vec = np.zeros((num_features,),dtype="float32")
    ehst_vec_set = set(ehst_model.index2word)
    count = 0
    for word in sentence:
        if word in ehst_vec_set:
            count = count + 1
            vector_w2vec_tfidf = ehst_model[word]
            feature_vec = np.add(feature_vec,vector_w2vec_tfidf)

    if count == 0:
        feature_vec = np.ones((num_features,),dtype="float32")
    else:
        feature_vec = np.divide(feature_vec,count)
    return feature_vec

def get_feature_vecs(clean_sentences,ehst_model,num_features):
    count = 0
    feature_vecs = np.zeros((len(clean_sentences),num_features),dtype="float32")
    for sentence in clean_sentences:
        feature_vecs[count] = make_feature_vec(sentence,ehst_model,num_features=300)
        count = count + 1
    return feature_vecs

def create_ngrams(text):
    vectorizer = CountVectorizer(ngram_range=(1,3))
    analyzer = vectorizer.build_analyzer()
    return analyzer(text)

print 'Loadin Word2Vec Model'
ehst_model = Word2Vec.load('gender_Word2vec_Vector')
print 'read training file'
train = read_data(training_file=training_file)

clean_sentences = []
for sentence in train['description']:
    sen = ' '.join(clean_obj.clean_article(sentence))
    clean_sentences.append(create_ngrams(sen))

print 'Started Featuring the Training Vectors'
train_data_vecs = get_feature_vecs(clean_sentences,ehst_model,num_features=300)
print 'started'
forest = RandomForestClassifier(n_estimators = 530,
                               max_depth = 10,
                               n_jobs = -1,
                               random_state = 00007,
                               criterion='gini',
                               verbose = 1)
forest = forest.fit(train_data_vecs,train["class"])
joblib.dump(forest,'gender_forest_model_530_10.pk',compress=3)


# forest = RandomForestClassifier(max_features='sqrt')
#
# parameter_grid = {
#                  'max_depth' : [4,5,6,7,8,10,12,15],
#                  'n_estimators': [200,300,250,420,350,530],
#                  'criterion': ['gini','entropy']
#                  }
#
# cross_validation = StratifiedKFold(train['class'], n_folds=5)
# grid_search = GridSearchCV(forest,
#                            param_grid=parameter_grid,
#                            cv=cross_validation)
#
# grid_search.fit(train_data_vecs, train['class'])
# print('Best score: {}'.format(grid_search.best_score_))
# print('Best parameters: {}'.format(grid_search.best_params_))


