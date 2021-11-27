import os
import kaggle
import requests

kaggle.api.authenticate() # for this to work, add the kaggle.json file to /Users/xxx/.kaggle folder
kaggle.api.dataset_download_files('arkhoshghalb/twitter-sentiment-analysis-hatred-speech', path='./data/kaggle_twitter', unzip=True)

if not os.path.exists("./data/github_twitter"):
    os.mkdir("./data/github_twitter")
os.system("curl -o ./data/github_twitter/train.csv https://github.com/t-davidson/hate-speech-and-offensive-language/raw/master/data/labeled_data.csv")

if not os.path.exists("./data/github_reddit"):
    os.mkdir("./data/github_reddit")
os.system("curl -o ./data/github_reddit/train.csv https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech/raw/master/data/reddit.csv")
