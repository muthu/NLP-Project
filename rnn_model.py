#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install tensorflow


# In[2]:


import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorboard.plugins import projector
from termcolor import colored
import numpy as np 
import pandas as pd
import random
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

import string
table = str.maketrans('', '', string.punctuation)


# In[3]:


def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(1235)
   tf.random.set_seed(1235)
   np.random.seed(1235)
   random.seed(1235)

reset_random_seeds()


# In[4]:


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


# In[5]:


twitter_df = pd.read_csv("data/clean/git_twitter.csv", index_col = "Unnamed: 0")
reddit_df = pd.read_csv("data/clean/reddit.csv", index_col = "Unnamed: 0")
reddit_df = reddit_df.dropna()


# In[6]:


num_to_sample = np.sum(reddit_df['Label']==1)
df_zero = reddit_df.query("Label==0").sample(n = num_to_sample, random_state=1)
df_one = reddit_df.query("Label==1")
reddit_df = df_zero.append(df_one, ignore_index=True)
reddit_df = reddit_df.sample(frac = 1)


# In[7]:


twitter_df['Data'] = twitter_df['Data'].str.lower()
reddit_df['Data'] = reddit_df['Data'].str.lower()
twitter_df['Data'] = twitter_df['Data'].apply(lambda x: ' '.join([word.translate(table) for word in x.split()]))
twitter_df['Data'] = twitter_df['Data'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
twitter_df['Data'] = twitter_df['Data'].apply(lambda x: ' '.join([porter.stem(word) for word in x.split()]))
reddit_df['Data'] = reddit_df['Data'].apply(lambda x: ' '.join([word.translate(table) for word in x.split()]))
reddit_df['Data'] = reddit_df['Data'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
reddit_df['Data'] = reddit_df['Data'].apply(lambda x: ' '.join([porter.stem(word) for word in x.split()]))


# In[8]:


(train, test) = train_test_split(reddit_df, test_size=0.2, random_state=42, shuffle=True)
(train, val) = train_test_split(train, test_size=0.2, random_state=42, shuffle=True)


# In[9]:


train_sentences = train['Data'].to_numpy()
test_sentences = test['Data'].to_numpy()
val_sentences = val['Data'].to_numpy()

train_labels = train['Label'].to_numpy()
test_labels = test['Label'].to_numpy()
val_labels = val['Label'].to_numpy()


# In[10]:


# vocab_size = 10000
oov_token = "<oov>"

tokeniser = Tokenizer(oov_token = oov_token)
tokeniser.fit_on_texts(train_sentences)
word_index = tokeniser.word_index
sequences = tokeniser.texts_to_sequences(train_sentences)
padding = pad_sequences(sequences,maxlen=120,truncating='post')

val_sequences = tokeniser.texts_to_sequences(val_sentences)
val_padded = pad_sequences(val_sequences,maxlen=120,truncating='post')

testing_sequences = tokeniser.texts_to_sequences(test_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=120,truncating='post')


# In[11]:


vocab_size = len(tokeniser.word_index) + 1


# In[12]:


simple_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size,16,input_length=120),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 10,activation="relu"),
    tf.keras.layers.Dense(units = 1, activation="sigmoid")
])


# In[13]:


es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1)
simple_model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
simple_model.fit(padding,train_labels,epochs = 10,validation_data=(val_padded,val_labels),callbacks=[EarlyStoppingAtMaxVal()])


# In[14]:


output = simple_model.evaluate(testing_padded,  test_labels, verbose=2)
y_preds = simple_model.predict(testing_padded)
pred_labels = np.where(y_preds > 0.5, 1, 0)
print(colored("The Simple NN model has a F1 score of: " + str(f1_score(test_labels, pred_labels)), 'green'))
print(colored("The Simple NN model gives us an accuracy of: " + str(output[1]), 'green'))


# In[15]:


import json
log_dir='SNN-model/'
if not os.path.exists(log_dir):
    os.system('mkdir SNN-model')

# Save Labels separately on a line-by-line manner.
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
  for subwords in json.loads(tokeniser.get_config()['word_counts']).keys():
    f.write("{}\n".format(subwords))

weights = tf.Variable(simple_model.layers[0].get_weights()[0][1:])
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Set up config.
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)


# In[16]:


rnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size,16,input_length=120),
    tf.keras.layers.SimpleRNN(units = 6, dropout=0.3, activation="tanh"),
    tf.keras.layers.Dense(units = 1, activation="sigmoid")
])


# In[17]:


rnn_model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
rnn_model.fit(padding,train_labels,epochs = 10,validation_data=(val_padded,val_labels),callbacks=[EarlyStoppingAtMaxVal()])


# In[18]:


output = rnn_model.evaluate(testing_padded,  test_labels, verbose=2)
y_preds = rnn_model.predict(testing_padded)
pred_labels = np.where(y_preds > 0.5, 1, 0)
print(colored("The RNN model has a F1 score of: " + str(f1_score(test_labels, pred_labels)), 'green'))
print(colored("The RNN model gives us an accuracy of: " + str(output[1]), 'green'))


# In[19]:


lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size,16,input_length=120),
    tf.keras.layers.LSTM(units = 6, dropout=0.3, activation="tanh"),
    tf.keras.layers.Dense(units = 1, activation="sigmoid")
])


# In[20]:


lstm_model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
lstm_model.fit(padding,train_labels,epochs = 10,validation_data=(val_padded,val_labels),callbacks=[EarlyStoppingAtMaxVal()])


# In[21]:


output = lstm_model.evaluate(testing_padded,  test_labels, verbose=2)
y_preds = lstm_model.predict(testing_padded)
pred_labels = np.where(y_preds > 0.5, 1, 0)
print(colored("The LSTM model has a F1 score of: " + str(f1_score(test_labels, pred_labels)), 'green'))
print(colored("The LSTM model gives us an accuracy of: " + str(output[1]), 'green'))


# In[22]:


bi_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size,16,input_length=120),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = 6, dropout=0.3, activation="tanh")),
    tf.keras.layers.Dense(units = 1, activation="sigmoid")
])


# In[23]:


bi_lstm_model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
bi_lstm_model.fit(padding,train_labels,epochs = 10,validation_data=(val_padded,val_labels),callbacks=[EarlyStoppingAtMaxVal()])


# In[24]:


output = bi_lstm_model.evaluate(testing_padded,  test_labels, verbose=2)
y_preds = bi_lstm_model.predict(testing_padded)
pred_labels = np.where(y_preds > 0.5, 1, 0)
print(colored("The Bi-LSTM model has a F1 score of: " + str(f1_score(test_labels, pred_labels)), 'green'))
print(colored("The Bi-LSTM model gives us an accuracy of: " + str(output[1]), 'green'))


# In[25]:


gru_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size,16,input_length=120),
    tf.keras.layers.GRU(units = 6, dropout=0.3, activation="tanh"),
    tf.keras.layers.Dense(units = 1, activation="sigmoid")
])


# In[26]:


gru_model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
gru_model.fit(padding,train_labels,epochs = 10,validation_data=(val_padded,val_labels),callbacks=[EarlyStoppingAtMaxVal()])


# In[27]:


output = gru_model.evaluate(testing_padded,  test_labels, verbose=2)
y_preds = gru_model.predict(testing_padded)
pred_labels = np.where(y_preds > 0.5, 1, 0)
print(colored("The GRU model has a F1 score of: " + str(f1_score(test_labels, pred_labels)), 'green'))
print(colored("The GRU model gives us an accuracy of: " + str(output[1]), 'green'))


# In[28]:


import json
log_dir='GRU-model/'
if not os.path.exists(log_dir):
    os.system('mkdir GRU-model')

# Save Labels separately on a line-by-line manner.
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
  for subwords in json.loads(tokeniser.get_config()['word_counts']).keys():
    f.write("{}\n".format(subwords))
  # Fill in the rest of the labels with "unknown".
#   for unknown in range(1, tokeniser.get_config()['num_words'] - len(encoder.subwords)):
#     f.write("unknown #{}\n".format(unknown))


# Save the weights we want to analyze as a variable. Note that the first
# value represents any unknown word, which is not in the metadata, here
# we will remove this value.
weights = tf.Variable(gru_model.layers[0].get_weights()[0][1:])
# Create a checkpoint from embedding, the filename and key are the
# name of the tensor.
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Set up config.
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)


# In[ ]:




