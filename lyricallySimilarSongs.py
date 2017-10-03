import pandas as pd
import scipy as sp
import numpy as np
from nltk import stem
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

#Indice of song
#0 to 57649
song_indice = 41221

songs = pd.read_csv('songdata.csv')

print('Finding songs most lyrically similar to...')
print('Artist: {}'.format(songs.iloc[song_indice,0]))
print('Song Title: {}'.format(songs.iloc[song_indice,1]))
print('Song Indice: {}'.format(song_indice))

print('Preprocessing lyrics...')

#Tokenize the lyrics
songs['text'] = songs.apply(lambda row: word_tokenize(str(row['text']).lower()),axis=1)

#Remove stop words (i.e. 'and', 'the', etc.)
stop_words = stopwords.words('english')
stop_words += string.punctuation
songs['text'] = songs.apply(lambda row: [x for x in row['text'] if not x in stop_words],axis=1)

#Stem words ('stopped' --> 'stop')
english_stemmer = stem.SnowballStemmer('english')
songs['text'] = songs.apply(lambda row: [english_stemmer.stem(x) for x in row['text']],axis=1)

#Convert list back to a single string
songs['text'] = songs.apply(lambda row: " ".join(row['text']).encode('utf-8') if row['text'] is not [] else "".encode('utf-8'),axis=1)

#Creates tf-idf vector from word count
vectorizer = TfidfVectorizer(min_df=1)
vectorized = vectorizer.fit_transform(songs['text'])

print('Clustering songs...')

#Clusters the songs so a given song only needs to have the distance calculated for the songs in it's cluster
num_clusters = 100
km = KMeans(n_clusters=num_clusters)
clustered = km.fit(vectorized)

#Predicts label for the given song
post_label = km.predict(vectorized[song_indice])[0]

#Retrieves all indices for the songs within the cluster
similar_indices = (km.labels_ == post_label).nonzero()[0]

#Calculates the distance between the given song and each of the songs within the cluster
similar = []
for i in similar_indices:
    dist = sp.linalg.norm((vectorized[song_indice] - vectorized[i]).toarray())
    similar.append((dist, songs.artist[i],songs.song[i],i))

similar = sorted(similar)

print('(Similarity, Artist, Song Title, Indice)')
print('Top ten most lyrically similar songs:')
print(similar[1:11])



