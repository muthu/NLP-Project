import os
import kaggle
import requests
import urllib.request
import wget

kaggle.api.authenticate() # for this to work, add the kaggle.json file to /Users/xxx/.kaggle folder
kaggle.api.dataset_download_files('arkhoshghalb/twitter-sentiment-analysis-hatred-speech', path='./data/kaggle_twitter', unzip=True)

if not os.path.exists("./data/github_reddit/train.csv"):
    os.mkdir("./data/github_reddit")
    url = 'https://raw.githubusercontent.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech/master/data/reddit.csv'
    wget.download(url,"./data/github_reddit/train.csv")


if not os.path.exists("./data/github_twitter/train.csv"):
    os.mkdir("./data/github_twitter")
    url = 'https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv'
    wget.download(url,"./data/github_twitter/train.csv")
