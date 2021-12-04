#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install tensorflow_hub


# In[2]:


import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import TimeDistributed
import tensorflow_hub as hub
import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
lem = nltk.stem.wordnet.WordNetLemmatizer()

tf.random.set_seed(1234)

import string
table = str.maketrans('', '', string.punctuation)


# In[3]:


class EarlyStoppingAtMaxVal(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(EarlyStoppingAtMaxVal, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_accuracy")
        if np.less(self.best, current):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            self.model.set_weights(self.best_weights)
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


# In[4]:


twitter_df = pd.read_csv("data/clean/git_twitter.csv", index_col = "Unnamed: 0")
reddit_df = pd.read_csv("data/clean/reddit.csv", index_col = "Unnamed: 0")
reddit_df = reddit_df.dropna()
reddit_df['Data'] = reddit_df['Data'].str.lower()
reddit_df['Data'] = reddit_df['Data'].apply(lambda x: ' '.join([word.translate(table) for word in x.split()]))
reddit_df['Data'] = reddit_df['Data'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
reddit_df['Data'] = reddit_df['Data'].apply(lambda x: ' '.join([porter.stem(word) for word in x.split()]))

# reddit_df['Data'] = reddit_df['Data'].apply(lambda x: ' '.join([lem.lemmatize(word) for word in x.split()]))


# In[5]:


# encoder = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
# encoder(['Hello World'])


# In[6]:


# reddit_df['Data'] = reddit_df['Data'].str.lower()
(train, test) = train_test_split(reddit_df, test_size=0.2, random_state=42, shuffle=True)
(train, val) = train_test_split(train, test_size=0.2, random_state=42, shuffle=True)


# In[7]:


train_sentences = train['Data'].to_numpy()
test_sentences = test['Data'].to_numpy()
val_sentences = val['Data'].to_numpy()

train_labels = train['Label'].to_numpy()
test_labels = test['Label'].to_numpy()
val_labels = val['Label'].to_numpy()


# In[8]:


vocab_size = 10000
model = tf.keras.models.Sequential()
model.add(hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4', 
                        input_shape=[], 
                        dtype=tf.string, 
                        trainable=False))
model.add(tf.keras.layers.Dense(6))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


# In[9]:


model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])


# In[10]:


model.fit(train_sentences, 
          train_labels, 
          epochs=10, 
          validation_data=(val_sentences, val_labels))


# In[11]:


token = Tokenizer()
token.fit_on_texts(reddit_df['Data'])
seq = token.texts_to_sequences(train_sentences)
padding = pad_sequences(seq,maxlen=300)

# token.fit_on_texts(val_sentenses)
seq = token.texts_to_sequences(val_sentences)
val_padded = pad_sequences(seq,maxlen=300)

# token.fit_on_texts(test_sentenses)
seq = token.texts_to_sequences(test_sentences)
test_padded = pad_sequences(seq,maxlen=300)


# In[ ]:





# In[12]:


vocab_size = len(token.word_index)+1


# In[13]:


# embedding_vector = {}
# f = open('glove.6B.300d.txt')
# for line in tqdm(f):
#     value = line.split(' ')
#     word = value[0]
#     coef = np.array(value[1:],dtype = 'float32')
#     embedding_vector[word] = coef


# In[14]:


# embedding_matrix = np.zeros((vocab_size,300))
# for word,i in tqdm(token.word_index.items()):
#     embedding_value = embedding_vector.get(word)
#     if embedding_value is not None:
#         embedding_matrix[i] = embedding_value


# In[15]:


# with open("embedding_mat_lem.npy","rb") as f:
#     embedding_matrix = np.load(f)
with open("embedding_mat.npy","rb") as f:
    embedding_matrix = np.load(f)


# In[16]:


simple_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size,300,weights = [embedding_matrix],input_length=300,trainable = True),
    tf.keras.layers.GRU(units = 6, dropout=0.3, activation="tanh"),
    tf.keras.layers.Dense(units = 1, activation="sigmoid")
])


# In[17]:


simple_model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
simple_model.fit(padding,train_labels,epochs = 10,validation_data=(val_padded,val_labels),callbacks=[EarlyStoppingAtMaxVal()])


# In[18]:


output = simple_model.evaluate(test_padded,  test_labels, verbose=2)
print(colored("The GloVe-GRU model gives us an accuracy of: " + str(output[1]), 'green'))

# In[ ]:





# In[ ]:




