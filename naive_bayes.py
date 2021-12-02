#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install sklearn')
# get_ipython().system('pip install nltk')


# In[50]:


# Code for creating Naive Bayes Classifier for textual data
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from termcolor import colored

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

import string
table = str.maketrans('', '', string.punctuation)


# In[51]:


twitter_df = pd.read_csv("data/clean/git_twitter.csv", index_col = "Unnamed: 0")
reddit_df = pd.read_csv("data/clean/reddit.csv", index_col = "Unnamed: 0")
reddit_df = reddit_df.dropna()


# # Cleaning Data using Stemming, removing stop words, removing punctuations etc.

# In[52]:


twitter_df['Data'] = twitter_df['Data'].str.lower()
reddit_df['Data'] = reddit_df['Data'].str.lower()
twitter_df['Data'] = twitter_df['Data'].apply(lambda x: ' '.join([word.translate(table) for word in x.split()]))
twitter_df['Data'] = twitter_df['Data'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
twitter_df['Data'] = twitter_df['Data'].apply(lambda x: ' '.join([porter.stem(word) for word in x.split()]))
reddit_df['Data'] = reddit_df['Data'].apply(lambda x: ' '.join([word.translate(table) for word in x.split()]))
reddit_df['Data'] = reddit_df['Data'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
reddit_df['Data'] = reddit_df['Data'].apply(lambda x: ' '.join([porter.stem(word) for word in x.split()]))


# In[55]:


merged_df = twitter_df.append(reddit_df, ignore_index="true")


# In[57]:


# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(sentences)
merged_df = merged_df.dropna()


# In[79]:


# (train, test) = train_test_split(merged_df, test_size=0.2, random_state=42, shuffle=True)
# (train, test) = train_test_split(twitter_df, test_size=0.2, random_state=42, shuffle=True)
(train, test) = train_test_split(reddit_df, test_size=0.2, random_state=42, shuffle=True)


# In[80]:


(train_X, train_y) = list(train["Data"]), list(train["Label"])
(test_X, test_y) = list(test["Data"]), list(test["Label"])


# In[81]:


model = make_pipeline(TfidfVectorizer(), BernoulliNB())


# In[82]:


model.fit(train_X, train_y)


# In[83]:


predicted_categories = model.predict(test_X)


# In[84]:


print(colored("The accuracy of Baseline Naive bayes is: " + str(accuracy_score(test_y, predicted_categories)), 'green'))


# In[85]:


(predicted_categories==1).sum()


# In[ ]:




