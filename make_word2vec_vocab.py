# -*- coding: utf-8 -*-
__author__ = "Raghav Pillai"

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from data_cleaning import CleanData
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

clean_obj = CleanData()
gender_file = "data_gender.csv"

def read_data(training_file):
    train = pd.read_csv(training_file,sep=',')
    train.dropna(axis=0,how='any')
    return train

def word_vec_train(sentences):
    model = Word2Vec(sentences=sentences,size=300,min_count=1,workers=8)
    model.init_sims(replace=True)
    model.save('gender_Word2vec_Vector')

def tfidf_vector(sentences):
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    vectorizer.fit_transform(sentences)
    idf = vectorizer.idf_
    word_vector_news = dict(zip(vectorizer.get_feature_names(), idf))
    from sklearn.externals import joblib
    joblib.dump(word_vector_news,'gender_TFIDF_Vector.pk',compress=3)

def create_ngrams(text):
    vectorizer = CountVectorizer(ngram_range=(1,3))
    analyzer = vectorizer.build_analyzer()
    return analyzer(text)


gender = read_data(gender_file)

sentences = gender['title']
print len(sentences)

total_sentences = []
total_tfidf_sentences = []
count = 0
for sentence in sentences:
    count += 1
    sen = ' '.join(clean_obj.clean_article(sentence))
    total_sentences.append(create_ngrams(sen))
    total_tfidf_sentences.append(' '.join(clean_obj.clean_article(sentence)))
    print count

word_vec_train(total_sentences)
tfidf_vector(total_tfidf_sentences)



