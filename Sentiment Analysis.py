import pandas as pd
import numpy as np
import re
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer 
wnl=WordNetLemmatizer()
from textblob import TextBlob
import string


publications=pd.read_csv(r"C:\Users\prana\Downloads\Anuja\New\t8.shakespeare.txt",sep='\t',names=['Text'])

stopwords=pd.read_csv(r"C:\Users\prana\Downloads\Anuja\New\stopwords.txt",sep='\t',names=['Words'])

publications.shape

def data_preprocessing(text):
    text=re.sub('<<THIS ELECTRONIC VERSION OF THE COMPLETE WORKS OF WILLIAM SHAKESPEARE IS COPYRIGHT 1990-1993 BY WORLD LIBRARY, INC., AND IS PROVIDED BY PROJECT GUTENBERG ETEXT OF ILLINOIS BENEDICTINE COLLEGE WITH PERMISSION. ELECTRONIC AND MACHINE READABLE COPIES MAY BE DISTRIBUTED SO LONG AS SUCH COPIES (1) ARE FOR YOUR OR OTHERS PERSONAL USE ONLY, AND (2) ARE NOT DISTRIBUTED OR USED COMMERCIALLY. PROHIBITED COMMERCIAL DISTRIBUTION INCLUDES BY ANY SERVICE THAT CHARGES FOR DOWNLOAD TIME OR FOR MEMBERSHIP.>>', ' ', text)
    text_tokens=word_tokenize(text)
    fil_text=[w for w in text_tokens if not w in set(stopwords['Words'])]
    return " ".join(fil_text)

publications.shape

publications['Text']=publications['Text'].apply(data_preprocessing)

publications=publications.drop_duplicates('Text')

publications.shape

def lemmatize(data):
    text=[wnl.lemmatize(word)for word in data]
    return data

publications['Text']=publications['Text'].apply(lambda x:lemmatize(x))

text= ' '.join([word for word in publications['Text']])
plt.figure(figsize=(14,12), facecolor='None')
wordcloud=WordCloud(max_words=500,width=1600,height=800).generate(text)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()



from nltk.tokenize import sent_tokenize
word_obj = {}
new_list =sent_tokenize(text)
for index,sentence in enumerate(new_list):
    word_count_per_sentence = len(sentence)
    word_obj[index] = word_count_per_sentence

Average_words_per_sentences = sum(word_obj.values())/len(word_obj)
Average_words_per_sentences

def polarity(text):
    return TextBlob(text).sentiment.polarity

def sentiment(label):
    if label<0:
        return "Negative"
    elif label==0:
        return "Neutral"
    elif label>0:
        return "Positive"
    

publications['polarity']=publications['Text'].apply(polarity)

publications['sentiment']=publications['polarity'].apply(sentiment)

fig=plt.figure(figsize=(12,8))
sns.countplot(x='sentiment',data=publications)



# Per Type Of Publication

# 1) Comedy

publications_comedy=pd.read_csv(r"C:\Users\prana\Downloads\Anuja\New\comedy.txt",sep='\t',names=['Text'])
stopwords=pd.read_csv(r"C:\Users\prana\Downloads\Anuja\New\stopwords.txt",sep='\t',names=['Words'])

def data_preprocessing(text):
    text=re.sub('<<THIS ELECTRONIC VERSION OF THE COMPLETE WORKS OF WILLIAM SHAKESPEARE IS COPYRIGHT 1990-1993 BY WORLD LIBRARY, INC., AND IS PROVIDED BY PROJECT GUTENBERG ETEXT OF ILLINOIS BENEDICTINE COLLEGE WITH PERMISSION. ELECTRONIC AND MACHINE READABLE COPIES MAY BE DISTRIBUTED SO LONG AS SUCH COPIES (1) ARE FOR YOUR OR OTHERS PERSONAL USE ONLY, AND (2) ARE NOT DISTRIBUTED OR USED COMMERCIALLY. PROHIBITED COMMERCIAL DISTRIBUTION INCLUDES BY ANY SERVICE THAT CHARGES FOR DOWNLOAD TIME OR FOR MEMBERSHIP.>>', ' ', text)
    text_tokens=word_tokenize(text)
    fil_text=[w for w in text_tokens if not w in set(stopwords['Words'])]
    return " ".join(fil_text)

publications_comedy['Text']=publications_comedy['Text'].apply(data_preprocessing)
publications_comedy=publications_comedy.drop_duplicates('Text')

def lemmatize(data):
    text=[wnl.lemmatize(word)for word in data]
    return data

publications_comedy['Text']=publications_comedy['Text'].apply(lambda x:lemmatize(x))

text_comedy= ' '.join([word for word in publications_comedy['Text']])

from nltk.tokenize import sent_tokenize
word_obj = {}
new_list =sent_tokenize(text_comedy)
for index,sentence in enumerate(new_list):
    word_count_per_sentence = len(sentence)
    word_obj[index] = word_count_per_sentence

Average_words_per_sentences_comedy = sum(word_obj.values())/len(word_obj)
Average_words_per_sentences_comedy

publications_comedy['polarity']=publications_comedy['Text'].apply(polarity)

publications_comedy['sentiment']=publications_comedy['polarity'].apply(sentiment)

fig=plt.figure(figsize=(12,8))
sns.countplot(x='sentiment',data=publications_comedy)



# History

publications_history=pd.read_csv(r"C:\Users\prana\Downloads\Anuja\New\history.txt",sep='\t',names=['Text'])
stopwords=pd.read_csv(r"C:\Users\prana\Downloads\Anuja\New\stopwords.txt",sep='\t',names=['Words'])

def data_preprocessing(text):
    text=re.sub('<<THIS ELECTRONIC VERSION OF THE COMPLETE WORKS OF WILLIAM SHAKESPEARE IS COPYRIGHT 1990-1993 BY WORLD LIBRARY, INC., AND IS PROVIDED BY PROJECT GUTENBERG ETEXT OF ILLINOIS BENEDICTINE COLLEGE WITH PERMISSION. ELECTRONIC AND MACHINE READABLE COPIES MAY BE DISTRIBUTED SO LONG AS SUCH COPIES (1) ARE FOR YOUR OR OTHERS PERSONAL USE ONLY, AND (2) ARE NOT DISTRIBUTED OR USED COMMERCIALLY. PROHIBITED COMMERCIAL DISTRIBUTION INCLUDES BY ANY SERVICE THAT CHARGES FOR DOWNLOAD TIME OR FOR MEMBERSHIP.>>', ' ', text)
    text_tokens=word_tokenize(text)
    fil_text=[w for w in text_tokens if not w in set(stopwords['Words'])]
    return " ".join(fil_text)

publications_history['Text']=publications_history['Text'].apply(data_preprocessing)
publications_history=publications_history.drop_duplicates('Text')

def lemmatize(data):
    text=[wnl.lemmatize(word)for word in data]
    return data

publications_history['Text']=publications_history['Text'].apply(lambda x:lemmatize(x))

text_history= ' '.join([word for word in publications_history['Text']])

from nltk.tokenize import sent_tokenize
word_obj = {}
new_list =sent_tokenize(text_history)
for index,sentence in enumerate(new_list):
    word_count_per_sentence = len(sentence)
    word_obj[index] = word_count_per_sentence

Average_words_per_sentences_history = sum(word_obj.values())/len(word_obj)
Average_words_per_sentences_history

publications_history['polarity']=publications_history['Text'].apply(polarity)

publications_history['sentiment']=publications_history['polarity'].apply(sentiment)

fig=plt.figure(figsize=(12,8))
sns.countplot(x='sentiment',data=publications_history)



# Sonnet

publications_sonnet=pd.read_csv(r"C:\Users\prana\Downloads\Anuja\New\THE SONNETS (1).txt",sep='\t',names=['Text'])
stopwords=pd.read_csv(r"C:\Users\prana\Downloads\Anuja\New\stopwords.txt",sep='\t',names=['Words'])

def data_preprocessing(text):
    text=re.sub('<<THIS ELECTRONIC VERSION OF THE COMPLETE WORKS OF WILLIAM SHAKESPEARE IS COPYRIGHT 1990-1993 BY WORLD LIBRARY, INC., AND IS PROVIDED BY PROJECT GUTENBERG ETEXT OF ILLINOIS BENEDICTINE COLLEGE WITH PERMISSION. ELECTRONIC AND MACHINE READABLE COPIES MAY BE DISTRIBUTED SO LONG AS SUCH COPIES (1) ARE FOR YOUR OR OTHERS PERSONAL USE ONLY, AND (2) ARE NOT DISTRIBUTED OR USED COMMERCIALLY. PROHIBITED COMMERCIAL DISTRIBUTION INCLUDES BY ANY SERVICE THAT CHARGES FOR DOWNLOAD TIME OR FOR MEMBERSHIP.>>', ' ', text)
    text_tokens=word_tokenize(text)
    fil_text=[w for w in text_tokens if not w in set(stopwords['Words'])]
    return " ".join(fil_text)

publications_sonnet['Text']=publications_sonnet['Text'].apply(data_preprocessing)
publications_sonnet=publications_sonnet.drop_duplicates('Text')

def lemmatize(data):
    text=[wnl.lemmatize(word)for word in data]
    return data

publications_sonnet['Text']=publications_sonnet['Text'].apply(lambda x:lemmatize(x))

text_sonnet= ' '.join([word for word in publications_sonnet['Text']])

from nltk.tokenize import sent_tokenize
word_obj = {}
new_list =sent_tokenize(text_sonnet)
for index,sentence in enumerate(new_list):
    word_count_per_sentence = len(sentence)
    word_obj[index] = word_count_per_sentence

Average_words_per_sentences_sonnet = sum(word_obj.values())/len(word_obj)
Average_words_per_sentences_sonnet

publications_sonnet['polarity']=publications_sonnet['Text'].apply(polarity)

publications_sonnet['sentiment']=publications_sonnet['polarity'].apply(sentiment)

fig=plt.figure(figsize=(12,8))
sns.countplot(x='sentiment',data=publications_sonnet)



# Tragedy

publications_tragedy=pd.read_csv(r"C:\Users\prana\Downloads\Anuja\New\tragedy.txt",sep='\t',names=['Text'])
stopwords=pd.read_csv(r"C:\Users\prana\Downloads\Anuja\New\stopwords.txt",sep='\t',names=['Words'])

def data_preprocessing(text):
    text=re.sub('<<THIS ELECTRONIC VERSION OF THE COMPLETE WORKS OF WILLIAM SHAKESPEARE IS COPYRIGHT 1990-1993 BY WORLD LIBRARY, INC., AND IS PROVIDED BY PROJECT GUTENBERG ETEXT OF ILLINOIS BENEDICTINE COLLEGE WITH PERMISSION. ELECTRONIC AND MACHINE READABLE COPIES MAY BE DISTRIBUTED SO LONG AS SUCH COPIES (1) ARE FOR YOUR OR OTHERS PERSONAL USE ONLY, AND (2) ARE NOT DISTRIBUTED OR USED COMMERCIALLY. PROHIBITED COMMERCIAL DISTRIBUTION INCLUDES BY ANY SERVICE THAT CHARGES FOR DOWNLOAD TIME OR FOR MEMBERSHIP.>>', ' ', text)
    text_tokens=word_tokenize(text)
    fil_text=[w for w in text_tokens if not w in set(stopwords['Words'])]
    return " ".join(fil_text)

publications_tragedy['Text']=publications_tragedy['Text'].apply(data_preprocessing)
publications_tragedy=publications_tragedy.drop_duplicates('Text')

def lemmatize(data):
    text=[wnl.lemmatize(word)for word in data]
    return data

publications_tragedy['Text']=publications_tragedy['Text'].apply(lambda x:lemmatize(x))

text_tragedy= ' '.join([word for word in publications_tragedy['Text']])

from nltk.tokenize import sent_tokenize
word_obj = {}
new_list =sent_tokenize(text_tragedy)
for index,sentence in enumerate(new_list):
    word_count_per_sentence = len(sentence)
    word_obj[index] = word_count_per_sentence

Average_words_per_sentences_tragedy = sum(word_obj.values())/len(word_obj)
Average_words_per_sentences_tragedy

publications_tragedy['polarity']=publications_tragedy['Text'].apply(polarity)

publications_tragedy['sentiment']=publications_tragedy['polarity'].apply(sentiment)

fig=plt.figure(figsize=(12,8))
sns.countplot(x='sentiment',data=publications_tragedy)

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords['Words']]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(publications['Text'], publications['sentiment'], test_size=0.25)


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier',RandomForestClassifier())])

## I selected RandomForestClassifier as my model, as the response variable is unbalanced, in such cases we can use 
## bagging technique to train our data

pipeline.fit(X_train,y_train)

predictions = pipeline.predict(X_test)

## This can be used to predict sentiments of any sentence
pipeline.predict(['This is the 100th Etext file presented by Project Gutenberg, and'])

predictions

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(predictions,y_test))

from sklearn.naive_bayes import MultinomialNB
pipeline_nb = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier',MultinomialNB())])

pipeline_nb.fit(X_train,y_train)

predictions_nb = pipeline_nb.predict(X_test)

print(classification_report(predictions_nb,y_test))
