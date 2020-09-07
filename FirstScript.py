from pyspark import SparkContext
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# In[5]:


access_token = "250650837-tl3Ae41KXq3qfqDhnBcacVAy0L0rW1kOc14fRZ1n"
access_token_secret = "8g34TXtscKYeQZohimreNqQTGhR8Zh5N5t1Px3oWyeWde"
consumer_key = "lPGYOIgYGSjmpdgut9awjKlpT"
consumer_secret = "FoJSBbuHN01iIkAvJzzkst8WGdrWcBQKUhLoe3M2FB7wz6PYKu"

# In[ ]:

'''
tweets_data = []
count = 0
lis = []
tags1 = []
element_number = 0
global printfunc


def printfunc(tweets_data):
    tag_dict = defaultdict(lambda: 0)
    Sum = 0
    print
    "Number of tweets from the begining:", count
    #         print "\n"
    for x in tweets_data:
        text = x['text']
        #       print text
        Sum += len(text)
        ent = x['entities']
        data = ent['hashtags']
        for x in data:
            tag = x['text']
            tag_dict[tag] += 1
    print
    "Top 5 hot hashtags:"
    c = 0
    for k, v in sorted(tag_dict.items(), reverse=True):
        if c < 5:
            print
            k, ":", v
            #                     print "\n"
            c += 1
    print
    "The Average Length of Tweet is:", (float(Sum / 100))
    print
    "\n"


#         print "\n"

class StdOutListener(StreamListener):

    def on_data(self, data):
        global count
        #         global Sum
        #         global element_number
        #         element_number += 1
        if (count < 100):
            count += 1
            #             print data
            tweet = json.loads(data)
            tweets_data.append(tweet)
        else:
            s = random.random()
            #             print count,"count"
            prob = (float(100) / float(count))
            #             print prob,"prob"
            if (s < prob):
                #                 data = tweets_data[1]
                S = int(s)
                tweet = json.loads(data)
                tweets_data[S] = tweet
                #                 print tweets_data[1]
                printfunc(tweets_data)
                count += 1
            else:
                count += 1
        return True

    def on_error(self, status):
        print
        "error", status


if __name__ == '__main__':
    tweet_list = []
    # This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    
'''


def get_tweets(username):
    # Authorization to consumer key and consumer secret
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

    # Access to user's access key and access secret
    auth.set_access_token(access_token, access_token_secret)

    # Calling api
    api = tweepy.API(auth)

    # 200 tweets to be extracted
    number_of_tweets = 200
    tweets = api.user_timeline(screen_name=username)

    # Empty Array
    tmp = []

    # create array of tweet information: username,
    # tweet id, date/time, text
    tweets_for_csv = [tweet.text for tweet in tweets]  # CSV file created
    for j in tweets_for_csv:
        # Appending tweets to the empty array tmp
        tmp.append(j)

        # Printing the tweets
    print(tmp)


# Driver code
if __name__ == '__main__':
    # Here goes the twitter handle for the user
    # whose tweets are to be extracted.
    get_tweets("twitter-handle")