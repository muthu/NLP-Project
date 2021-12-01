#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import re


# In[2]:


kt_df = pd.read_csv("data/kaggle_twitter/train.csv")
gt_df = pd.read_csv("data/github_twitter/train.csv")
gr_df = pd.read_csv("data/github_reddit/train.csv")


# # Cleaning Reddit Data and storing in CSV 

# In[20]:


gr_df['hate_speech_idx'] = gr_df['hate_speech_idx'].fillna(0)
data = list()
label = list()
for index, row in gr_df.iterrows():
    if row["hate_speech_idx"] != 0:
        ini_list = row["hate_speech_idx"]
        res = ini_list.strip('][').split(', ')
        res = [int(item) - 1 for item in res]
        temp = row["text"].replace("\t","").split("\n")
        main = [x[x.find('.') + 2:] for x in temp]
        main = main[:-1]
        main = [x.strip("'") for x in main]
        main = [x.strip('"') for x in main]
        main = [x.lstrip('>') for x in main]
        res = [x for x in res if x < len(main)]
        data = data + [main[x] for x in res]
        label = label + [1 for x in res]
        notres = [x for x in range(len(main)) if x not in res]
        data = data + [main[x] for x in notres]
        label = label + [0 for x in notres]
        
dictionary = {'Label':label, 'Data':data}
train_data = pd.DataFrame(dictionary)
if not os.path.exists("data/clean"):
    os.system("mkdir data/clean")
train_data.to_csv("data/clean/reddit.csv")


# # Cleaning Twitter Kaggle Data and Storing in CSV

# In[15]:


clean_tweets = []
clean_labels = []
for index, row in kt_df.iterrows():
    s = re.sub(r'[^a-zA-Z0-9_!@#$%^&*\(\)-= \{\}\[\]:;\"]', '', row['tweet'])
    s = re.sub('bihday','birthday',s)
    s = s.strip(" ")
    s = s.strip("\t")
    clean_tweets.append(s)
    clean_labels.append(row["label"])
    
tw_kg = {"Label": clean_labels, "Data": clean_tweets}
train_data = pd.DataFrame(tw_kg)
if not os.path.exists("data/clean"):
    os.system("mkdir data/clean")
train_data.to_csv("data/clean/git_twitter.csv")


# In[14]:





# In[ ]:




