from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_combination(n):

    combinations = [] 

    for i in range(n):

        cols = [
            "new_title",
            "overview",
            "spoken_languages",
            "original_language",
            "production_companies",
            "production_countries"
        ]  

        for i in range(len(cols)):
            random_number = random.randint(1, 10)
            for x in range(random_number):
                cols.append(cols[i])
        
        combinations.append(cols)

    return combinations


def recommend_movies(df, movie_title, cosine_sim, n):

  # get the index of the movie that matches the title
  idx = df[df["title"] == movie_title].index[0]

  # create a Series with the similarity scores in descending order
  score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

  lst = []

  # add 5 most similar movies to the list
  for i in list(score_series.iloc[1:n].index):
    lst.append([df.iloc[i]["title"], score_series[i]])

  return lst


# TfidfVectorizer --> convert a collection of text documents to a matrix of TF-IDF features
#- Welche Gewichtung hat das Wort im Overview --> df_tfidf_vect

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
    

# returns a dataframe that contains each movie with its 10 top recommended movies and scores
def recommendations_final(df, movie_lst, cols):

    # initialize the new dataframe
    df_movies_recommended = pd.DataFrame()
        
    # add cols to column tfidf
    df["tfidf"] = df[cols].apply(" ".join, axis=1)

    # create cosine similarity matrix
    cosine_sim = create_cosine(df["tfidf"])

    # loop through all movies
    for movie in movie_lst:   

        # get 10  movie recommendations for each movie based on cosine similarity matrix
        recommended_movies = recommend_movies(df, movie, cosine_sim, 6)  

        # add column combination, movie, recommended movies and scores to dataframe
        df_movies_recommended = df_movies_recommended.append({
            "movie": movie,
            "recommended_movie_1":  recommended_movies[0][0],
            "score_1": recommended_movies[0][1], 
            "recommended_movie_2": recommended_movies[1][0],
            "score_2": recommended_movies[1][1],
            "recommended_movie_3": recommended_movies[2][0],
            "score_3": recommended_movies[2][1],
            "recommended_movie_4": recommended_movies[3][0],
            "score_4": recommended_movies[3][1],
            "recommended_movie_5": recommended_movies[4][0],
            "score_5": recommended_movies[4][1],
        }, ignore_index = True)

    return df_movies_recommended


# returns a dataframe that contains each movie with its 10 top recommended movies and scores
def recommendations(df, df_movies):

    # initialize the new dataframe
    df_movies_recommended = pd.DataFrame()

    # loop through 20 combinations of columns
    for combination in create_combination(20):
        
        # add cols to column tfidf
        df["tfidf"] = df[combination].apply(" ".join, axis=1)

        # create cosine similarity matrix
        cosine_sim = create_cosine(df["tfidf"])

        # loop through all movies
        for movie in df_movies["title"]:   

            # get 10  movie recommendations for each movie based on cosine similarity matrix
            recommended_movies = recommend_movies(df, movie, cosine_sim, 11)  

            # add ccolumn combination, movie, recommended movies and scores to dataframe
            df_movies_recommended = df_movies_recommended.append({
                "combination": combination, 
                "movie": movie,
                "recommended_movie_1":  recommended_movies[0][0],
                "score_1": recommended_movies[0][1], 
                "recommended_movie_2": recommended_movies[1][0],
                "score_2": recommended_movies[1][1],
                "recommended_movie_3": recommended_movies[2][0],
                "score_3": recommended_movies[2][1],
                "recommended_movie_4": recommended_movies[3][0],
                "score_4": recommended_movies[3][1],
                "recommended_movie_5": recommended_movies[4][0],
                "score_5": recommended_movies[4][1],
                "recommended_movie_6": recommended_movies[5][0],
                "score_6": recommended_movies[5][1],
                "recommended_movie_7": recommended_movies[6][0],
                "score_7": recommended_movies[6][1],
                "recommended_movie_8": recommended_movies[7][0],
                "score_8": recommended_movies[7][1],
                "recommended_movie_9": recommended_movies[8][0],
                "score_9": recommended_movies[8][1],
                "recommended_movie_10": recommended_movies[9][0],
                "score_10": recommended_movies[9][1]
            }, ignore_index = True)

    return df_movies_recommended