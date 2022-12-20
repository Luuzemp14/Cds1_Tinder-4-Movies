import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import random
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

''' STEMMING TOKENIZER '''
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
''''''
''' CREATE COMBINATIONS FOR TFIDF '''
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
''''''

''' LSA FINAL FUNCTION '''
def lsa_final(liked, movies, cols):
    # Erstellen Sie eine Liste mit den Titeln und Beschreibungen aller Filme
    # create tfidf column with column combination

    movies["LSA"] = movies[cols].apply(" ".join, axis=1)
    titles = movies['title'].tolist()
    descriptions = movies['LSA'].tolist()

    # Verwenden Sie TfidfVectorizer, um die Titel und Beschreibungen der Filme in Vektoren umzuwandeln
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(descriptions)

    # Führen Sie LSA durch, indem Sie TruncatedSVD verwenden
    svd = TruncatedSVD(n_components=100)
    latent_vectors = svd.fit_transform(vectors)
    sorted_similarities = []

    for like in liked:
        # Finden Sie den Film, dessen Titel dem gegebenen Titel entspricht
        if like not in titles:
            print('Film not found in DataFrame')
        else:
            index = titles.index(like)
            if index >= len(movies):
                print('Index not found in DataFrame')
            else:
                movie_vector = latent_vectors[index]

        # Berechnen Sie die Cosinus-Ähnlichkeit zwischen dem Film und allen anderen Filmen

        movie_vector = movie_vector.reshape(1, -1)
        latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], -1)
        similarities = cosine_similarity(movie_vector, latent_vectors)

        # sort similarity scores and get names of movies
        sorted_similarities.append(similarities[0][similarities[0].argsort()[::-1]])

    # flatten sorted_similarities list
    sorted_similarities = [item for sublist in sorted_similarities for item in sublist]

    sorted_indexes = [i for i in similarities[0].argsort()[::-1]][1:6]

    sorted_movies = [titles[i] for i in sorted_indexes]

    return sorted_movies
''''''
''' DOC2VEC FINAL FUNCTION '''
def tagged_document(list_of_list_of_words, titles):
   for i, list_of_words in enumerate(list_of_list_of_words):
      yield gensim.models.doc2vec.TaggedDocument(list_of_words.split(), [titles[i]])
''''''
''' TFIDF FINAL FUNCTION '''
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


# returns a dataframe that contains each movie with its 10 top recommended movies and scores

def recommendations_tfidf_final(df, movie_lst, cols):

  # initialize the new dataframe
  lst = []

  # add cols to column tfidf
  df["tfidf"] = df[cols].apply(" ".join, axis=1)

  # create cosine similarity matrix
  cosine_sim = create_cosine(df["tfidf"])

  # loop through all movies
  for movie in movie_lst:
      # get 10  movie recommendations for each movie based on cosine similarity matrix
      recommended_movies = recommend_movies(df, movie, cosine_sim, 11)
      # add ccolumn combination, movie, recommended movies and scores to dataframe
      lst.append(recommended_movies)

  lst = [item for sublist in lst for item in sublist]
  lst = sorted(lst, key=lambda x: x[1], reverse=True)
  # remove score from lst but keep the order
  lst = [item[0] for item in lst]

  return lst[:5]
''''''
'''RECOMMENTATIONS PER COMBINATION'''
# returns a dataframe that contains each movie with its 10 top recommended movies and scores
def recommendations_per_comb(df, df_movies):

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
''''''
''' TFIDF ON USERPROFILE'''

def tfidf_rec_user(df, split, cols):
    # create tfidf column with column combination
    df["tfidf"] = df[cols].apply(" ".join, axis=1)

    # create cosine similarity matrix
    cosine_sim = create_cosine(df["tfidf"])

    testing_movies = {}

    for key, values in split.items():
        movie_lst = []
        for value in values:
            recommended_movies = recommend_movies(df, value, cosine_sim, 11)
            movie_lst.append(recommended_movies)

        # flatten list
        movie_lst = [item for sublist in movie_lst for item in sublist]

        # sort list on score
        movie_lst = sorted(movie_lst, key=lambda x: x[1], reverse=True)

        # remove the score from movie list but keep the order
        movie_lst = [item[0] for item in movie_lst]

        # remove duplicates but keep the order
        movie_lst = list(dict.fromkeys(movie_lst))

        testing_movies[key] = movie_lst[:5]

    return testing_movies
''''''

'''LSA ON USERPROFILE'''
def lsa_rec_user(df, split, cols):

    df["LSA"] = df[cols].apply(" ".join, axis=1)
    titles = df['title'].tolist()
    descriptions = df['LSA'].tolist()
    # Verwenden Sie TfidfVectorizer, um die Titel und Beschreibungen der Filme in Vektoren umzuwandeln
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(descriptions)
    # Führen Sie LSA durch, indem Sie TruncatedSVD verwenden
    svd = TruncatedSVD(n_components=100)
    latent_vectors = svd.fit_transform(vectors)
    testing_movies = {}
    for key, values in split.items():
        movie_lst = []

        for value in values:

        # Finden Sie den Film, dessen Titel dem gegebenen Titel entspricht

            if value not in titles:
                print('Film not found in DataFrame')
            else:
                index = titles.index(value)
                if index >= len(df):
                    print('Index not found in DataFrame')
                else:
                    movie_vector = latent_vectors[index]
        # Berechnen Sie die Cosinus-Ähnlichkeit zwischen dem Film und allen anderen Filmen

            movie_vector = movie_vector.reshape(1, -1)

            latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], -1)

            similarities = cosine_similarity(movie_vector, latent_vectors)

            movie_lst.append(similarities[0][similarities[0].argsort()[::-1]])

        # flatten list
        movie_lst = [item for sublist in movie_lst for item in sublist]
        # sort list on score
        movie_lst = [i for i in similarities[0].argsort()[::-1]][1:6]

        # remove duplicates but keep the order
        movie_lst = [titles[i] for i in movie_lst]
        # remove where there is a movie multiple times
        testing_movies[key] = movie_lst[:20]
    return testing_movies
''''''

'''DOC2VEC ON USERPROFILE'''
def doc2vec_rec_user(df, split, cols):
    # create tfidf column with column combination
    df["doc2vec"] = df[cols].apply(" ".join, axis=1)

    tagged_document = list(tagged_document(df["doc2vec"], df['movieId']))

    model = Doc2Vec(tagged_document, vector_size=90, window=5, min_count=1, workers=4)

    testing_movies = {}

    count = 0
    for key, values in split.items():
        count += 1
        print(count)
        movie_lst = []
        for value in values:
            # get id of movie title
            movie_id = df[df["title"] == value]["movieId"].values[0]

            # from tagged_documents get words where tags = movie_id
            word_list = [word for word, tag in tagged_document if tag == movie_id]

            # flatten list
            word_list = [item for sublist in word_list for item in sublist]

            # infer vector of movie overview
            word_list_vectorized = model.infer_vector(word_list)

            # get cosine similarity of movie overview with all other movie overviews
            similar_sentences = model.docvecs.most_similar([word_list_vectorized], topn = 6)

            # sort by score
            similar_sentences = sorted(similar_sentences, key = lambda x: x[1], reverse = True)

            # exclude 1st recommendation (movie itself)
            similar_sentences = similar_sentences[1:]

            # get movie ids of similar sentences
            similar_sentences = [movie_id for movie_id, score in similar_sentences]

            # get movie titles of similar sentences
            recommendations = df[df["movieId"].isin(similar_sentences)]["title"].values

            movie_lst.append(recommendations)

        # flatten list
        movie_lst = [item for sublist in movie_lst for item in sublist]

        # sort list on score
        movie_lst = sorted(movie_lst, key=lambda x: x[1], reverse=True)

        # remove the score from movie list but keep the order
        movie_lst = [item[0] for item in movie_lst]

        testing_movies[key] = movie_lst[:5]

    return testing_movies
''''''
'''Ploting Hits and precision'''
def calculate_hits(data):
    user = 0
    precision = []
    for key, values in test_movies.items():
        
        for value in values:
            ## check if value is in recommended movies and at which position
            if value in data[key]:
                hits += 1
                precision.append(data[key].index(value))
    
    plot_hits(precision, hits)

def plot_hits(precision, hits):

    # count number of hits per position (dictionary)
    c = {i:precision.count(i) for i in precision}

    plt.figure(figsize=(10, 8))
    plt.bar(c.keys(), c.values())
    plt.title("Precision of Hits (n Hits: {})".format(hits))
    plt.xlabel("Position")
    plt.ylabel("Number of Hits")
    plt.show()
''''''


'''Count Hits and precision'''
def calculate_hits(recs,test_set):
    total_hits = 0
    users = 0
    precision = []
    for key, values in test_set.items():
        hits_per_user = 0
        for value in values:
            ## check if value is in recommended movies and at which position
            if value in recs[key]:
                hits_per_user += 1
                total_hits +=1
                precision.append(recs[key].index(value))
        if hits_per_user >0:
            users += 1
    plot_hits(precision, total_hits,users,len(test_set))

def plot_hits(precision, hits,user,users):

    # count number of hits per position (dictionary)
    c = {i:precision.count(i) for i in precision}

    plt.figure(figsize=(10, 8))
    plt.bar(c.keys(), c.values())
    plt.title("Precision of Hits (n Hits: {})".format(hits)+",found on {} Users".format(user)+" out of {} Users".format(users))
    plt.xlabel("Position")
    plt.ylabel("Number of Hits")
    plt.show()
''''''