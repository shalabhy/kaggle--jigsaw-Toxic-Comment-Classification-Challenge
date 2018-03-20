# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

import re
def cleantext(x): 
    x = re.sub(r'[^\w\s]','',x) 
    x = re.sub(r'[0-9]','',x)
    x = re.sub(r'["\n"]'," ",x)
    x = x.lower()
    return x

train["comment_text"]= train["comment_text"].fillna("empty")
test["comment_text"]= test["comment_text"].fillna("empty")

train["comment_text"]= train["comment_text"].apply(cleantext)
test["comment_text"]= test["comment_text"].apply(cleantext)

raw_text = np.hstack((train["comment_text"],test["comment_text"]))

from keras.preprocessing import text
token = text.Tokenizer(num_words= 100000)
token.fit_on_texts(raw_text)
train['comment_text'] = token.texts_to_sequences(train['comment_text'])
test['comment_text'] = token.texts_to_sequences(test['comment_text'])

max_words_train = max(train["comment_text"].apply(lambda x: len(x)))
max_words_test = max(test["comment_text"].apply(lambda x: len(x)))
max_words = max(max_words_train,max_words_test)
raw = pd.concat([train["comment_text"],test["comment_text"]])
vocab = 100000
# Any results you write to the current directory are saved as output.
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
np.random.seed(42)

X_train = sequence.pad_sequences(train["comment_text"], maxlen=max_words+1)
X_test = sequence.pad_sequences(test["comment_text"], maxlen=max_words+1)
trainx = X_train[:150000]
validx = X_train[150000:]
trainy = train[['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']][:150000]
validy = train[['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']][150000:]

model = Sequential()
model.add(Embedding(vocab, 32, input_length=max_words+1))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

sub = pd.read_csv("../input/sample_submission.csv")
    
for i in ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']:

    model.fit(trainx, trainy[i], validation_data=(validx, validy[i]), epochs=2, batch_size=1000, verbose=2)
    pred = model.predict(X_test)
    sub[i] = pred

sub.to_csv("keras_embedding.csv",index = False)    

# lgbm

from sklearn.feature_extraction.text import TfidfVectorizer

word_vectorizer = TfidfVectorizer(stop_words = 'english',analyzer = 'word',ngram_range = (1,1),max_features = 10000)
word_vectorizer.fit(raw_text)

train_vec_train = word_vectorizer.transform(train['comment_text'][:150000])
y_train = train[['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']][:150000]
train_vec_valid = word_vectorizer.transform(train['comment_text'][150000:])
y_valid = train[['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']][150000:]
test_vec = word_vectorizer.transform(test['comment_text'])

char_vectorizer = TfidfVectorizer(stop_words = 'english',analyzer = 'char',ngram_range = (5,6),max_features = 10000)
char_vectorizer.fit(raw_text)

train_vec_char = char_vectorizer.transform(train['comment_text'][:150000])
valid_vec_char = char_vectorizer.transform(train['comment_text'][150000:])
test_vec_char = char_vectorizer.transform(test['comment_text'])

from scipy import sparse

train_final = sparse.hstack([train_vec_train,train_vec_char])
valid_final = sparse.hstck([train_vec_valid,valid_vec_char])
test_final = sparse.hstack([test_vec,test_vec_char])

sub2 = pd.read_csv("../input/sample_submission.csv")
import lightgbm
parameters = {
    'metric': 'auc',
    'learning_rate': 0.05,
    'verbose': 1}

for i in ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']:
    train_data = lightgbm.Dataset(train_final, label= y_train[i])
    valid_data = lightgbm.Dataset(valid_final, label = y_valid[i])
    model = lightgbm.train(parameters, train_data, valid_sets=valid_data, num_boost_round=300,early_stopping_rounds=50)
    pred = model.predict(test_final)
    sub2[i] = pred
    sub2[i] = 0.5* (sub[i]+sub2[i])                  
    
    
sub2.to_csv("final_solution.csv",index = False)    
