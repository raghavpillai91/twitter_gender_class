# -*- coding: utf-8 -*-
__author__ = "Rahul Babu R"

#importing url library for fetching url
import urllib2

#importing base64 package
import base64

#importing command line parser
import argparse

#importing JSON
import json

#importing urllib for urlencoding
import urllib

##########################Keys for Collection##########################
with open('raghav.json') as forpro_file:
    crawler_pw = json.load(forpro_file)

consumer_key = crawler_pw['twitter_consumer_k']
consumer_secret = crawler_pw['twitter_consumer_sec']
#######################################################################

API_VERSION = '1.1'
API_ENDPOINT = 'https://api.twitter.com'
REQUEST_TOKEN_URL = '%s/oauth2/token' % API_ENDPOINT
SEARCH_API ='%s/%s/search/tweets.json' % (API_ENDPOINT,API_VERSION)

class Twitter(object):

    def __init__(self, parser, token=None):

        # parsing arguments from commandline
        arguments = parser.parse_args()

        #initializing the consumer key of the applicaiton
        self.consumerKey = consumer_key

        #initializing the consumer secret of the twitter app
        self.consumerSecret = consumer_secret

        #initializing the accessToken of the twitter app
        self.accessToken = ''

        #initializing the accessTokenSecret of the twitter app
        self.accessTokenSecret = ''

        #initializing the query to search
        self.q = ' '.join(arguments.q)

        #initializing the resultCount
        # 100
        self.count = '100'

        #the language of tweet
        self.lang = 'en'

        #initializing the resultType
        self.resultType = 'mixed'

        #connecting to the twitter
        if token:
            self._token = token
        else:
            self._token = self.connectTwitter()

    def connectTwitter(self):
        '''
        connect to twiter api end-point https://api.twitter.com/oauth2/token
        and obtain an oauth token
        '''
        bearer_token = '%s:%s' % (self.consumerKey, self.consumerSecret)
        encoded_bearer_token = base64.b64encode(bearer_token.encode('ascii'))
        request = urllib2.Request(REQUEST_TOKEN_URL)
        request.add_header('Content-Type', 'application/x-www-form-urlencoded;charset=UTF-8')
        request.add_header('Authorization', 'Basic %s' % encoded_bearer_token.decode('utf-8'))
        request.add_data('grant_type=client_credentials'.encode('ascii'))

        try:
            response = urllib2.urlopen(request)
        except urllib2.HTTPError, e:
            print 'HTTPError = ' + str(e.code)
        except urllib2.URLError, e:
            print 'URLError = ' + str(e.reason)
        except Exception:
            import traceback
            print 'generic exception: ' + traceback.format_exc()

        rawData = response.read().decode('utf-8')
        data = json.loads(rawData)
        return data['access_token']

    def make_url(self,next_results=None):

        if not next_results:
            params = {'result_type': self.resultType, 'q': self.q, 'count': self.count, 'lang': self.lang}
            paramsEncode = urllib.urlencode(params)
            full_url = "%s?%s" % (SEARCH_API, paramsEncode)
        else:
            full_url = "%s%s" % (SEARCH_API, str(next_results))

        return full_url

    def requestApi(self,full_url):
        request = urllib2.Request(full_url)
        request.add_header('Authorization', 'Bearer %s' % self._token)
        try:
            response = urllib2.urlopen(request)
        except urllib2.HTTPError, e:
            print 'HTTPError = ' + str(e.code)
        except urllib2.URLError, e:
            print 'URLError = ' + str(e.reason)
        except Exception:
            import traceback
            print 'generic exception: ' + traceback.format_exc()

        rawData = response.read().decode('utf-8')
        data = json.loads(rawData)
        return data

    def makeDataDictionary(self,data):

        #initialize a list contains Twitter Data
        twitterDataDictList = []
        #check any data is there
        if 'statuses' in data:

            #iterate through each data
            for status in data['statuses']:

                #initialize a blank dictionary to store twitter data
                twitterDataDict = {}
                #initialize the tweet text if it is there
                if 'text' in status:
                    twitterDataDict['Description'] = status['text']
                    twitterDataDict['title'] = status['text']
                else:
                    twitterDataDict['Description'] = None
                    twitterDataDict['title'] = None

                if 'user' in status:

                    if 'name' in status['user']:

                        twitterDataDict['Author'] = status['user']['name']
                    else:
                        twitterDataDict['Author'] = None

                twitterDataDictList.append(twitterDataDict)
        return twitterDataDictList


    def searchTwitter(self):

        full_url = self.make_url()

        data = self.requestApi(full_url=full_url)

        twitterDataDictList = self.makeDataDictionary(data=data)

        if 'search_metadata' in data:

            if 'next_results' in data['search_metadata']:

                next_results = data['search_metadata']['next_results']

            else:
                return twitterDataDictList

        for page in range(1,20):

            full_url = self.make_url(next_results=next_results)

            try:
                data = self.requestApi(full_url=full_url)

                temp_twitterDataDictList = self.makeDataDictionary(data=data)

                twitterDataDictList += temp_twitterDataDictList
            except:
                return twitterDataDictList

            if 'search_metadata' in data:
                if 'next_results' in data['search_metadata']:

                    next_results = data['search_metadata']['next_results']
                else:
                    return twitterDataDictList



        return twitterDataDictList


def main():
    # create object for the parser
    parser = argparse.ArgumentParser()

    # create option to read the query string
    parser.add_argument('-q', '--q', help='Query String to Search in Twitter', nargs='+')

    try:
        twitterObj = Twitter(parser=parser)
        twitterDataDictList = twitterObj.searchTwitter()
        import pandas as pd
        df = pd.DataFrame(twitterDataDictList)
        df.to_csv('twitterDataDictList.csv', sep=',', index=False, encoding='utf-8')
    except Exception, e:
        print e.message
        print 'Twitter Exception'
if __name__ == '__main__':
    main()