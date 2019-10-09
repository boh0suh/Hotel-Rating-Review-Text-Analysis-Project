# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:29:56 2019

@author: Boh Young Suh

---
Code below is to compare how similar two plots are from imdb and rotten tomatoes
for a given movie. Both plots are combined into one to generate a full tfidf matrix.
Cosine similarity is used to calculate the distance between imdb plot vector 
and rotten tomatoes plot vector for each given movie. Movies that do not have plots
either from imdb and rotten tomatoes are removed from calculation.

"""

# imports
import pandas as pd
import numpy as np
import re

import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# read in plot file
plots = pd.read_csv('../data/imdb_rt_plots.csv', sep='|')

# remove all null values in either plot 
plots = plots[plots.rt_plot.notnull() & plots.imdb_plot.notnull()]
plots = plots.reset_index()

# preprocess plot text to remove stop words, special characters, etc.
def preprocess(sentences):
    output = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence)
        stop_words = stopwords.words('english')
        stemmer = EnglishStemmer()
        filtered_words = [w for w in tokens if not w in stop_words]
        filtered_words = [''.join([c for c in w if not c.isdigit()]) for w in filtered_words]
        filtered_words = [' '.join([w for w in filtered_words if len(w)>2])]
        filtered_words = [w for w in filtered_words if not w in stop_words]
        filtered_words = [stemmer.stem(w) for w in filtered_words]
        
        output.append("".join(filtered_words))
    return np.array(output)

# preprocess imdb and rt plots
plots['rt_cleaned_plot'] = pd.Series(preprocess(plots['rt_plot'].values))
plots['imdb_cleaned_plot'] = pd.Series(preprocess(plots['imdb_plot'].values))

# convert each plot columns to list and combine 
joined_plot = plots.rt_cleaned_plot.tolist() + plots.imdb_cleaned_plot.tolist()

# compute tf-idf matrix
tfidf_vectorizer=TfidfVectorizer(analyzer='word',sublinear_tf=True,stop_words='english',norm='l2')

# apply to joined plots
combined_plots_vector=tfidf_vectorizer.fit_transform(joined_plot)

# convert results to dataframe
combined_plots = pd.DataFrame(combined_plots_vector.toarray())

# total number of movies count
movie_counts = int(len(combined_plots)/2)

# compute cosine similarity between imdb and rotten tomatoes plot
from sklearn.metrics.pairwise import cosine_similarity

scores = []
#count = 0
for i in range(0,movie_counts):

    cossim = cosine_similarity(combined_plots.iloc[[i]], combined_plots.iloc[[i+movie_counts]])
    scores.append(cossim[0][0])

plots['similarity score'] = scores  

plots = plots[['dossier_id', 'similarity score']]

# file output
plots.to_csv('../results/movie_plot_similarity_scores.csv', sep='|', header=True, index=False)
