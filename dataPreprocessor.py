from gensim import models
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np
from sklearn.externals import joblib


class preaperData:
    def create_wodr2vec_model(self,sentences):
        num_features = 120
        Word2vecModel = preaperData.create_word2vec(sentences)
        model_name = "model_trainData1"
        Word2vecModel.save(model_name)

    def crete_data_vec(self,sentences):
        num_features=120
        Word2vecModel = models.Word2Vec.load("model_trainData1")
        DataVecs = preaperData.getAvgFeatureVecs(sentences, Word2vecModel, num_features)
        return DataVecs

    @staticmethod
    def create_word2vec(train_data):
        num_features = 120
        min_word_count = 0
        num_workers = 4
        context = 10
        downsampling = 1e-3
        model = models.Word2Vec(train_data, workers=num_workers, \
                                  size=num_features, min_count=min_word_count, \
                                  window=context, sample=downsampling)
        model.init_sims(replace=True)
        return model

    @staticmethod
    def create_ngrams(text):
        vectorizer = CountVectorizer(ngram_range=(1,3))
        analyzer = vectorizer.build_analyzer()
        return analyzer(text)

    def clean_data(self,file):

        read_data = pd.read_csv(file, sep=',')
        preprocessed_data = [[''] * 2 for i in range(len(read_data))]
        for i in range(len(read_data)):
            remove_html_tags = BeautifulSoup(read_data["Description"][i])
            letters_only = re.sub("[^a-zA-Z]", " ", remove_html_tags.get_text())
            lower_case = letters_only.lower()
            words = lower_case.split()
            words = [w for w in words if not w in stopwords.words("english")]
            preprocessed_data[i][0]=" ".join(words)
            preprocessed_data[i][1]=read_data["Class"][i]
            sentences = []
            threat_class = []
            for i in preprocessed_data:
                sentences.append(preaperData.create_ngrams(i[0]))
                threat_class.append(i[1])
        return sentences,threat_class
    @staticmethod
    def getAvgFeatureVecs(reviews, model, num_features):
        counter = 0.
        reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
        for review in reviews:
            reviewFeatureVecs[counter] = preaperData.makeFeatureVec(review, model,num_features)
            counter = counter + 1.
        return reviewFeatureVecs


    @staticmethod
    def makeFeatureVec(words, model, num_features):
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0.
        index2word_set = set(model.index2word)
        for word in words:
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec, model[word])
        featureVec = np.divide(featureVec, nwords)
        return featureVec
    def predict_input(self,testDataVec,test_class):
        clf = joblib.load('engine_max_ensemble.pkl')
        result=clf.predict(testDataVec)
        print result
    def makeTestData(self,sentences):
        num_features = 120
        Word2vecModel = preaperData.create_word2vec(sentences)
        DataVecs = preaperData.getAvgFeatureVecs(sentences, Word2vecModel, num_features)
        return DataVecs
