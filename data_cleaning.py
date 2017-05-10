# -*- coding: utf-8 -*-

#importing HTMLParser to unescape html characters
import HTMLParser

#importing nltk for stopwords
from nltk.corpus import stopwords

#importing regular expression module
import re

#importing CSV Module to read csv files
import csv

#importing Beautifulsoup to clean html
from bs4 import BeautifulSoup

import itertools

STOPWORDS = stopwords.words("english")

with open('slang.csv', 'rb') as csv_file:
    reader = csv.reader(csv_file)
    SLANGS = dict(reader)

APOSTROPHE = {"'s":"is","'re":"are"}

class CleanData(object):

    @staticmethod
    def clean_article(article):

        #unescaping the HTML Characters
        _html_parser = HTMLParser.HTMLParser()

        try:
            _cleaned_article = _html_parser.unescape(str(article))
        except:
            _cleaned_article = str(article)

        #Remove HTML
        _cleaned_article = BeautifulSoup(_cleaned_article,"html.parser").get_text()

        #change apostrophe normalization
        words = _cleaned_article.split()
        reformed = [APOSTROPHE[word] if word in APOSTROPHE else word for word in words]
        _cleaned_article = ' '.join(reformed)

        #removing URLS and Special Characters
        _cleaned_article =re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', _cleaned_article)
        _cleaned_article= re.compile("[^\w']|_").sub(' ',_cleaned_article)
        _cleaned_article = re.sub("\d+", "", _cleaned_article)

        #decoding to UTF-8
        try:
            _cleaned_article = _cleaned_article.decode('utf8').encode('ascii','ignore')
        except:
            pass

        #removing stopwords
        _cleaned_article  = ' '.join([word for word in _cleaned_article.split() if word not in STOPWORDS])

        #split attached words eg:- SunnyNight - Sunny Night
        # _cleaned_article = ' '.join(re.findall('[A-Z][^A-Z]*', _cleaned_article))

        #change slang words to normal english words
        words = _cleaned_article.split()
        reformed = [SLANGS[word] if word in SLANGS else word for word in words]
        _cleaned_article = ' '.join(reformed)

        #standardizing words
        _cleaned_article = ''.join(''.join(s)[:2] for _, s in itertools.groupby(_cleaned_article))

        #remove short words
        shortword = re.compile(r'\W*\b\w{1,2}\b')
        _cleaned_article = shortword.sub('',_cleaned_article).strip()

        _cleaned_article = _cleaned_article.lower().split()

        return _cleaned_article










