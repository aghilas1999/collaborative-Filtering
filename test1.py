import os
import time
import gc
import argparse

# Data science imports
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

# Utils import
from fuzzywuzzy import fuzz

class UserBasedRecommender:
    def __init__(self, path_movies, path_ratings):
        self.path_movies = path_movies
        self.path_ratings = path_ratings
        self.movie_rating_thres = 0
        self.user_rating_thres = 0

    def set_filter_params(self, movie_rating_thres, user_rating_thres):
        self.movie_rating_thres = movie_rating_thres
        self.user_rating_thres = user_rating_thres

    def _prep_data(self):
        # Read data
        df_movies = pd.read_csv(os.path.join(self.path_movies),
                               usecols=['movieId', 'title'],
                               dtype={'movieId': 'int32', 'title': 'str'})
        df_ratings = pd.read_csv(os.path.join(self.path_ratings),
                                usecols=['userId', 'movieId', 'rating'],
                                dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

        # Filter data
        df_movies_cnt = pd.DataFrame(df_ratings.groupby('movieId').size(),
                                     columns=['count'])
        popular_movies = list(set(df_movies_cnt.query('count >= @self.movie_rating_thres').index))
        movies_filter = df_ratings.movieId.isin(popular_movies).values

        df_users_cnt = pd.DataFrame(df_ratings.groupby('userId').size(),
                                    columns=['count'])
        active_users = list(set(df_users_cnt.query('count >= @self.user_rating_thres').index))
        users_filter = df_ratings.userId.isin(active_users).values

        df_ratings_filtered = df_ratings[movies_filter & users_filter]

        # Pivot and create user-movie matrix
        user_movie_mat = df_ratings_filtered.pivot(index='userId', columns='movieId', values='rating').fillna(0)

        return user_movie_mat

    def _fuzzy_matching(self, hashmap, fav_user):
        match_tuple = []
        for user in hashmap:
            ratio = fuzz.ratio(user.lower(), fav_user.lower())
            if ratio >= 60:
                match_tuple.append((user, ratio))
        match_tuple = sorted(match_tuple, key=lambda x: x[1], reverse=True)
        if not match_tuple:
            print('Oops! No match is found.')
        else:
            print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
            return match_tuple[0][0]

    def _inference(self, data, fav_user_id, n_recommendations):
        print('You have input user ID:', fav_user_id)
        # Calculate Pearson similarity
        similar_users = pd.DataFrame(1 - pairwise_distances(data, data[fav_user_id].values.reshape(1, -1), metric='correlation'))
        similar_users.columns = ['similarity']
        similar_users = similar_users[similar_users.index != fav_user_id]

        # Get top N similar users
        similar_users = similar_users.sort_values(by='similarity', ascending=False)
        top_users = similar_users.head(n_recommendations)

        # Get movies that the top N similar users have watched and the target user has not watched
        watched_movies = data[fav_user_id]
        recommended_movies = []
        for index, row in top_users.iterrows():
            user = data.loc[index]
            unrated_movies = user[user.isnull()].index  # Movies not rated by this user
            for movie in unrated_movies:
                if not pd.notnull(watched_movies[movie]):
                    recommended_movies.append(movie)

        return recommended_movies

    def make_recommendations(self, fav_user_id, n_recommendations):
        user_movie_mat = self._prep_data()
        recommended_movies = self._inference(user_movie_mat, fav_user_id, n_recommendations)

        if recommended_movies:
            print('Recommendations for user ID {0}:'.format(fav_user_id))
            for i, movie in enumerate(recommended_movies, start=1):
                print('{0}. Movie ID: {1}'.format(i, movie))


def parse_args():
    parser = argparse.ArgumentParser(
        prog="User-Based Movie Recommender",
        description="Run User-Based Movie Recommender")
    parser.add_argument('--path', nargs='?', default='../data/MovieLens',
                        help='input data path')
    parser.add_argument('--movies_filename', nargs='?', default='movies.csv',
                        help='provide movies filename')
    parser.add_argument('--ratings_filename', nargs='?', default='ratings.csv',
                        help='provide ratings filename')
    parser.add_argument('--user_name', nargs='?', default='',
                        help='provide the username for recommendations')
    parser.add_argument('--top_n', type=int, default=10,
                        help='top n movie recommendations')
    return parser.parse_args()

'''
if __name__ == '__main__':
    args = parse_args()
    data_path = ""
    movies_filename = "movies.csv"
    ratings_filename = "rating.csv"
    user_id = args.user_id  # Remplacez par l'ID de l'utilisateur
    top_n = args.top_n

    recommender = UserBasedRecommender(
        os.path.join(data_path, movies_filename),
        os.path.join(data_path, ratings_filename))

    recommender.set_filter_params(50, 50)

    recommender.make_recommendations(user_id, top_n)
'''
if __name__ == '__main__':
    # Définissez les chemins vers les fichiers directement ici
    path_movies = 'movies.csv'  # Remplacez par le chemin réel vers le fichier movies.csv
    path_ratings = 'ratings.csv'  # Remplacez par le chemin réel vers le fichier ratings.csv
    movie_name = 1  # Le film pour lequel vous souhaitez des recommandations
    top_n = 10  # Le nombre de recommandations que vous souhaitez obtenir

    # initial recommender system
    recommender = UserBasedRecommender(path_movies, path_ratings)
    # set params
    recommender.set_filter_params(50, 50)
    #recommender.set_model_params(20, 'brute', 'cosine', -1)
    # make recommendations
    recommender.make_recommendations(movie_name, top_n)