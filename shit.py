from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_movies(df, movie_title, cosine_sim):

  # get the index of the movie that matches the title
  idx = df[df["title"] == movie_title].index[0]

  # create a Series with the similarity scores in descending order
  score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

  lst = []

  for i in list(score_series.iloc[1:11].index):
    lst.append([df.iloc[i]["title"], score_series[i]])

  return lst
  

def create_cosine(data):
    # create tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    # transform data to tfidf matrix (sparse matrix)
    tfidf = tfidf_vectorizer.fit_transform(data)

    cosine_sim = cosine_similarity(tfidf)
    
    return cosine_sim


def stemming_tokenizer(data):
  # create stemmer object
  porter_stemmer = PorterStemmer()

  # define stop words
  stop_words = set(stopwords.words('english'))

  overviews = []
    
  for overview in data:
    # remove all non-alphabetic characters
    overview = re.sub(r'[^a-zA-Z]', ' ', overview)

    # remove single words
    overview = re.sub(r'\b[a-zA-Z]\b', ' ', overview)

    # overview to lowercase
    overview = overview.lower()

    # split overview into words
    words = overview.split()

    # remove stop words from words
    words = [w for w in words if not w in stop_words]

    # stem words
    words = [porter_stemmer.stem(word) for word in words]

    # join words to build 1 string sentence
    words_filtered = " ".join(words)

    overviews.append(words_filtered)

  return overviews

    