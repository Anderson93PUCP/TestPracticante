
# coding: utf-8

# In[1]:

#Importamos la siguientes librerias

import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')
import sklearn
import sklearn.datasets
import sklearn.linear_model
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[8]:

#Cargamos nuestra base de datos 
data_set = pd.read_csv('tweets.csv')


# In[9]:

#Verificamos las columnas vacias
data_set.isnull().sum()


# In[10]:

#Eliminamos las columnas nulas
del data_set["airline_sentiment_gold"]
del data_set["negativereason_gold"]
del data_set["negativereason_confidence"]


# In[11]:

#Seleccionamos los tweets positivosy negativos para nuestra TFIDF usaremos solo 10 tweets
tweets=data_set.loc[(data_set["airline_sentiment"]=="positive") | (data_set["airline_sentiment"]=="negative")]
textdata=tweets[["text","airline_sentiment"]].head(10)
textdata


# In[12]:

#Eliminamos caracteres inecesarios
textdata=textdata['text'].str.replace("^@\\w+ *", "", case=True)


# In[13]:

textdata


# In[14]:

#Cada tweet o texto sera colocado en alldocuments 
alldocuments=[]
for document in textdata.values:
    alldocuments.append(document)


# In[15]:

#Aplicamos una implementacion basica de tfidf para calcular que palabras son las mas importantes en los 10 tweets o documentos
#y lo calculamos mediante la funcion sklearn_tfidf.fit_transform
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True)
sklearn_representation = sklearn_tfidf.fit_transform(alldocuments)


# In[16]:

#Mostramos la informacion
sklearn_representation.data


# In[17]:

#Para la clasificacion bayesiana me tome la libertad de utilizar la libreria nltk, puesto que tengo mas dominio utilizando
#esta libreria y la otra estoy en un proceso de aprendizaje aun
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk.metrics
#Las columnas seleccionadas son los tweets y el sentimiento que puede ser positivo o negativo
tweetsclass=tweets[["text","airline_sentiment"]] 
tweetsclass=tweetsclass.reset_index(drop=True) #Reseteamos los indices
#Separamos nuestros tweets de prueba y de entrenamiento
tweets2=tweetsclass[6000:8000]
tweets=tweetsclass.head(5000)


# In[31]:

#Esta funcion me permite obtener todas las palabras de los tweets a analizar en una lista
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

#Esta funcion agrupa las palabras con distancias mas cortas por su frecuencia
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


def extract_features(names):
    document_words = set(names)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


# In[40]:

#Clasificacion bayesiana
tw=[]
for i in range(len(tweets["text"])):
    #tokenizer
    w=word_tokenize(tweets["text"][i])
    tw.append((w,tweets["airline_sentiment"][i]))

    
word_features = get_word_features(get_words_in_tweets(tw))


# In[24]:

#obtenemos nuestro conjunto de datos para entrenar el modelo
training_set = nltk.classify.apply_features(extract_features, tw)


# In[25]:

#Aplicamos el metodo de clasificacion bayesiana
classifier = nltk.NaiveBayesClassifier.train(training_set)
#Predecimos un tweet cualquiera
tweet = 'I had a good flight'
print (classifier.classify(extract_features(tweet.split())))


# In[ ]:



