#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install tensorflow


# In[71]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 
import pandas as pd

from termcolor import colored

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

tf.random.set_seed(1234)

import string
table = str.maketrans('', '', string.punctuation)

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


# In[72]:


twitter_df = pd.read_csv("data/clean/git_twitter.csv", index_col = "Unnamed: 0")
reddit_df = pd.read_csv("data/clean/reddit.csv", index_col = "Unnamed: 0")
reddit_df = reddit_df.dropna()


# In[73]:


twitter_df['Data'] = twitter_df['Data'].str.lower()
reddit_df['Data'] = reddit_df['Data'].str.lower()
twitter_df['Data'] = twitter_df['Data'].apply(lambda x: ' '.join([word.translate(table) for word in x.split()]))
twitter_df['Data'] = twitter_df['Data'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
twitter_df['Data'] = twitter_df['Data'].apply(lambda x: ' '.join([porter.stem(word) for word in x.split()]))
reddit_df['Data'] = reddit_df['Data'].apply(lambda x: ' '.join([word.translate(table) for word in x.split()]))
reddit_df['Data'] = reddit_df['Data'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
reddit_df['Data'] = reddit_df['Data'].apply(lambda x: ' '.join([porter.stem(word) for word in x.split()]))


# In[42]:


(train, test) = train_test_split(reddit_df, test_size=0.2, random_state=42, shuffle=True)
(train, val) = train_test_split(train, test_size=0.2, random_state=42, shuffle=True)


# In[45]:


train_sentences = train['Data'].to_numpy()
test_sentences = test['Data'].to_numpy()
val_sentences = val['Data'].to_numpy()

train_labels = train['Label'].to_numpy()
test_labels = test['Label'].to_numpy()
val_labels = val['Label'].to_numpy()


# In[48]:


vocab_size = 10000
oov_token = "<oov>"

tokeniser = Tokenizer(num_words = vocab_size,oov_token = oov_token)
tokeniser.fit_on_texts(train_sentences)
word_index = tokeniser.word_index
sequences = tokeniser.texts_to_sequences(train_sentences)
padding = pad_sequences(sequences,maxlen=120,truncating='post')

val_sequences = tokeniser.texts_to_sequences(val_sentences)
val_padded = pad_sequences(val_sequences,maxlen=120,truncating='post')

testing_sequences = tokeniser.texts_to_sequences(test_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=120,truncating='post')


# In[74]:


simple_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size,16,input_length=120),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 10,activation="relu"),
    tf.keras.layers.Dense(units = 1, activation="sigmoid")
])


# In[75]:


simple_model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
simple_model.fit(padding,train_labels,epochs = 10,validation_data=(val_padded,val_labels),callbacks=[EarlyStoppingAtMaxVal()])


# In[76]:


output = simple_model.evaluate(testing_padded,  test_labels, verbose=2)
print(colored("The simple NN model gives us an accuracy of: " + str(output[1]), 'green'))


# In[81]:


rnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size,16,input_length=120),
    tf.keras.layers.SimpleRNN(units = 6, dropout=0.3, activation="tanh"),
    tf.keras.layers.Dense(units = 1, activation="sigmoid")
])


# In[82]:


rnn_model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
rnn_model.fit(padding,train_labels,epochs = 10,validation_data=(val_padded,val_labels),callbacks=[EarlyStoppingAtMaxVal()])


# In[83]:


output = rnn_model.evaluate(testing_padded,  test_labels, verbose=2)
print(colored("The Baseline RNN model gives us an accuracy of: " + str(output[1]), 'green'))


# In[77]:


lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size,16,input_length=120),
    tf.keras.layers.LSTM(units = 6, dropout=0.3, activation="tanh"),
    tf.keras.layers.Dense(units = 1, activation="sigmoid")
])


# In[78]:


lstm_model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
lstm_model.fit(padding,train_labels,epochs = 10,validation_data=(val_padded,val_labels),callbacks=[EarlyStoppingAtMaxVal()])


# In[79]:


output = lstm_model.evaluate(testing_padded,  test_labels, verbose=2)
print(colored("The LSTM model gives us an accuracy of: " + str(output[1]), 'green'))


# In[84]:


bi_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size,16,input_length=120),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = 6, dropout=0.3, activation="tanh")),
    tf.keras.layers.Dense(units = 1, activation="sigmoid")
])


# In[85]:


bi_lstm_model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
bi_lstm_model.fit(padding,train_labels,epochs = 10,validation_data=(val_padded,val_labels),callbacks=[EarlyStoppingAtMaxVal()])


# In[86]:


output = bi_lstm_model.evaluate(testing_padded,  test_labels, verbose=2)
print(colored("The BI-LSTM model gives us an accuracy of: " + str(output[1]), 'green'))


# In[90]:


gru_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size,16,input_length=120),
    tf.keras.layers.GRU(units = 6, dropout=0.3, activation="tanh"),
    tf.keras.layers.Dense(units = 1, activation="sigmoid")
])


# In[91]:


gru_model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
gru_model.fit(padding,train_labels,epochs = 10,validation_data=(val_padded,val_labels),callbacks=[EarlyStoppingAtMaxVal()])


# In[92]:


output = gru_model.evaluate(testing_padded,  test_labels, verbose=2)
print(colored("The GRU model gives us an accuracy of: " + str(output[1]), 'green'))


# In[ ]:




