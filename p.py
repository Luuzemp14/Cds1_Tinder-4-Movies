import re
import random
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def stemming_tokenizer(data):
    """
    Stemming Tokenizer function to split a sentence or document into individual tokens
    or "words" and then reduce each token to its base form.
    
    Parameters:
        data (list): list of overviews

    Returns:
        overviews (list): list of stemmed overviews
    """

    # create stemmer object
    porter_stemmer = PorterStemmer()

    # define stop words
    stop_words = set(stopwords.words("english"))

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


def create_combination(n):
    """
    Create a list of lists with column combinations

    Parameters:
        n (int): number of combinations

    Returns:
        combinations (list): list of lists with column combinations
    """

    combinations = [] 

    for i in range(n):

        # create list with column names
        cols = [
            "new_title",
            "overview",
            "spoken_languages",
            "original_language",
            "production_companies",
            "production_countries"
        ]  

        for i in range(len(cols)):

            # create random number between 1 and 10
            random_number = random.randint(1, 10)

            # append column * random number of times to list
            for x in range(random_number):
                cols.append(cols[i])
        
        combinations.append(cols)

    return combinations


def create_cosine(data):
    # create tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    # transform data to tfidf matrix (sparse matrix)
    tfidf = tfidf_vectorizer.fit_transform(data)

    # calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf)
  
    return cosine_sim


def create_tagged_document(list_of_list_of_words, titles):
    """
    Creates a tagged document for each movie overview with the movie title as tag.

    Parameters:
        list_of_overviews (list): list of all overviews
        titles (list): list of titles

    Returns:
        list: list of tagged documents - example: [TaggedDocument(words=['hello', 'world'], tags=[1753])]
    """

    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words.split(), [titles[i]])


def recommend_movies(df, movie_title, cosine_sim, n):
    """
    Recommends movies based on a cosine similarity matrix

    Parameters:
        df (DataFrame): DataFrame with movies
        movie_title (str): title of movie
        cosine_sim (array): cosine similarity matrix
        n (int): number of recommendations

    Returns:
        lst (list): list of recommended movies
   """
    
    # get the index of the movie that matches the title
    idx = df[df["title"] == movie_title].index[0]

    # create a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    lst = []

    # add 5 most similar movies to the list
    for i in list(score_series.iloc[1:n].index):
        lst.append([df.iloc[i]["title"], score_series[i]])

    return lst


def recommendations_per_comb(df, df_movies):
    """
    Creates a tfidf model and recommends 10 movies per movie per combination of columns

    Parameters:
        df (DataFrame): Stemmed dataFrame with movies
        df_movies (DataFrame): DataFrame with movies

    Returns:
        df_movies_recommended (DataFrame): DataFrame with recommended movies and scores
    """

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


def tfidf_final(movie_lst, df, cols):
    """
    Creates a tfidf model and recommends movies based on similar movies

    Parameters:
        df (DataFrame): DataFrame with movies
        movie_lst (list): list of liked movies
        cols (list): list of columns

    Returns:
        lst (list): list of recommended movies
    """

    lst = []

    # create tfidf column with column combination
    df["tfidf"] = df[cols].apply(" ".join, axis=1)

    # create cosine similarity matrix
    cosine_sim = create_cosine(df["tfidf"])

    # loop over all movies
    for movie in movie_lst:
        # get 10  movie recommendations for each movie based on cosine similarity matrix
        recommended_movies = recommend_movies(df, movie, cosine_sim, 11)
        lst.append(recommended_movies)

    # if movie_lst is longer than 1, get first 3 recommendations for each movie
    if len(movie_lst) > 1:
        # get first 3 recommendations for each movie
        lst = [item[:3] for item in lst]

    # flatten lst
    lst = [item for sublist in lst for item in sublist]

    # sort lst based on score
    lst = sorted(lst, key=lambda x: x[1], reverse=True)

    # remove score from lst but keep the order
    lst = [item[0] for item in lst]

    return lst[:5]


def lsa_final(liked, movies, cols):
    """
    Creates a latent semantic analysis (LSA) model and recommends movies based on similar movies

    Parameters:
        liked (list): list of liked movies
        movies (DataFrame): DataFrame with movies
        cols (list): list of columns

    Returns:
        sorted_movies (list): list of recommended movies
    """

    # create tfidf column with column combination
    movies["LSA"] = movies[cols].apply(" ".join, axis=1)

    # create list with titles and list with descriptions
    titles = movies['title'].tolist()
    descriptions = movies['LSA'].tolist()

    # create tfidf vectorizer
    vectorizer = TfidfVectorizer()

    # transform data to tfidf matrix
    vectors = vectorizer.fit_transform(descriptions)

    # create svd object
    svd = TruncatedSVD(n_components = 100)

    # latent vectors 
    latent_vectors = svd.fit_transform(vectors)

    sorted_similarities = []

    # loop over liked movies and get movie vector with index
    for like in liked:
        if like not in titles:
            print("Film not found in DataFrame")
        else:
            index = titles.index(like)
            if index >= len(movies):
                print("Index not found in DataFrame")
            else:
                movie_vector = latent_vectors[index]

        # reshape movie_vector and latent_vectors that they have the same shape (dimension)
        movie_vector = movie_vector.reshape(1, -1)
        latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], -1)

        # calculate cosine similarity between movie_vector and latent_vectors
        similarities = cosine_similarity(movie_vector, latent_vectors)

        # sort similarity scores and get names of movies
        sorted_similarities.append(similarities[0][similarities[0].argsort()[::-1]])

    # if movies are longer than 1, get first 3 recommendations for each movie
    if len(liked) > 1:
        # get first 3 recommendations for each movie
        sorted_similarities = [item[:3] for item in sorted_similarities]

    # flatten sorted_similarities list
    sorted_similarities = [item for sublist in sorted_similarities for item in sublist]

    # sort similarities and get top 5 movies
    sorted_indexes = [i for i in similarities[0].argsort()[::-1]][1:6]

    # get titles of top 5 movies
    sorted_movies = [titles[i] for i in sorted_indexes]

    return sorted_movies


def doc2vec_final(df, movies, cols):
    df["D2V"] = df[cols].apply(" ".join, axis=1)

    # call function to create tagged document
    tagged_documents = list(create_tagged_document(df["D2V"], df["movieId"]))

    # create doc2vec model
    model = Doc2Vec(tagged_documents, vector_size = 90, window = 5, min_count = 2, workers = 4)

    lst = []

    # loop over all movies
    for movie in movies:
        movie_id = df[df["title"] == movie]["movieId"].values[0]

        # from tagged_documents get words where tags = movie_id
        word_list = [word for word, tag in tagged_documents if tag == movie_id]

        # flatten list
        word_list = [item for sublist in word_list for item in sublist]

        # infer vector of movie overview
        word_list_vectorized = model.infer_vector(word_list)

        # get cosine similarity of movie overview with all other movie overviews
        similar_sentences = model.docvecs.most_similar([word_list_vectorized], topn = 11)

        # sort by score
        similar_sentences = sorted(similar_sentences, key = lambda x: x[1], reverse = True)

        # exclude 1st recommendation (movie itself)
        similar_sentences = similar_sentences[1:]

        lst.append(similar_sentences)

    # if movie_lst is longer than 1, get first 3 recommendations for each movie
    if len(movies) > 1:
        # get first 3 recommendations for each movie
        lst = [item[:3] for item in lst]

    # flatten lst
    lst = [item for sublist in lst for item in sublist]

    # sort lst based on score
    lst = sorted(lst, key=lambda x: x[1], reverse = True)

    # get movie ids of similar sentences
    movie_ids = [item[0] for item in lst]

    # get movie titles of movie_ids
    recommendations = df[df["movieId"].isin(movie_ids)]["title"].values

    return recommendations[:5]
    

def tfidf_rec_user(df, split, cols):
    """
    Creates a tfidf model and recommends movies based on the user profile.

    Parameters:
        df (dataframe): dataframe with movies
        split (dict): dictionary with user_id and list of movies
        cols (list): list of columns to use for tfidf
    
    Returns:
        testing_movies (dict): dictionary with user_id and list of recommended movies
    """

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


def lsa_rec_user(df, split, cols):
    """
    Creates a latent semantic analysis model and recommends movies based on the user profile.

    Parameters:
        df (dataframe): dataframe with movies
        split (dict): dictionary with user_id and list of movies
        cols (list): list of columns to use for lsa
    
    Returns:
        testing_movies (dict): dictionary with user_id and list of recommended movies
    """

    # create lsa column with column combination
    df["LSA"] = df[cols].apply(" ".join, axis=1)

    # create list of titles and descriptions
    titles = df["title"].tolist()
    descriptions = df["LSA"].tolist()

    # create tfidf vectorizer   
    vectorizer = TfidfVectorizer()

    # transform data to tfidf matrix
    vectors = vectorizer.fit_transform(descriptions)

    # create svd object
    svd = TruncatedSVD(n_components = 100)

    # latent vectors 
    latent_vectors = svd.fit_transform(vectors)

    testing_movies = {}

    for key, values in split.items():
        movie_lst = []
        for value in values:
            if value not in titles:
                print("Film not found in DataFrame")
            else:
                index = titles.index(value)
                if index >= len(df):
                    print("Index not found in DataFrame")
                else:
                    movie_vector = latent_vectors[index]

            # reshape movie_vector and latent_vectors that they have the same shape (dimension)
            movie_vector = movie_vector.reshape(1, -1)
            latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], -1)

            # calculate cosine similarity between movie_vector and latent_vectors
            similarities = cosine_similarity(movie_vector, latent_vectors)

            # sort similarity scores and get names of movies
            movie_lst.append(similarities[0][similarities[0].argsort()[::-1]])

        # flatten list
        movie_lst = [item for sublist in movie_lst for item in sublist]
        # sort list on score
        movie_lst = [i for i in similarities[0].argsort()[::-1]][1:6]

        # remove duplicates but keep the order
        movie_lst = [titles[i] for i in movie_lst]

        testing_movies[key] = movie_lst[:20]

    return testing_movies


def doc2vec_rec_user(df, split, cols):
    """
    Creates a doc2vec model and recommends movies based on the user profile.

    Parameters:
        df (dataframe): dataframe with movies
        split (dict): dictionary with user_id and list of movies
        cols (list): list of columns to use for doc2vec
    
    Returns:
        testing_movies (dict): dictionary with user_id and list of recommended movies
    """

    # create tfidf column with column combination
    df["doc2vec"] = df[cols].apply(" ".join, axis=1)

    # create tagged documents
    tagged_document = list(create_tagged_document(df["doc2vec"], df["movieId"]))

    model = Doc2Vec(tagged_document, vector_size=90, window=5, min_count=1, workers=4)

    testing_movies = {}

    for key, values in split.items():
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


def plot_hits(precision, hits, users, total_users):
    """
    Plot the precision of hits

    Parameters:
        precision (list): list with the position of the hit
        hits (int): total number of hits
        user (int): number of users with at least one hit
        users (int): total number of users
    """
    # count number of hits per position (dictionary)
    c = {i:precision.count(i) for i in precision}

    # plot hits
    plt.figure(figsize=(8, 5))
    plt.bar(c.keys(), c.values())
    plt.title("Precision of Hits (n Hits: {})\n found on {} Users out of a total of {} users".format(hits, users, total_users))
    plt.xlabel("Position")
    plt.ylabel("Number of Hits")
    plt.show()


def calculate_hits(recs,test_set):
    """
    Calculate the number of hits and precision for each user and the total number of hits and precision

    Parameters:
        recs (dict): dictionary with recommendations for each user
        test_set (dict): dictionary with test set for each user

    Output:
        total_hits (int): total number of hits
    """

    total_hits = 0
    users = 0
    precision = []

    for key, values in test_set.items():
        hits_per_user = 0
        if key not in recs:
            continue
        for value in values:
            ## check if value is in recommended movies and at which position
            if value in recs[key]:
                hits_per_user += 1
                total_hits +=1
                precision.append(recs[key].index(value))   

        # increament users if user has at least one hit 
        if hits_per_user > 0:
            users += 1
    
    # call plot function
    plot_hits(precision, total_hits, users, len(test_set))