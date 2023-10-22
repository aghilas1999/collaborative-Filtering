import pandas as pd
import numpy as np
from scipy.spatial.distance import cityblock

# Chargez les données depuis les fichiers CSV
movies_dataframe = pd.read_csv("movies.csv")
ratings_dataframe = pd.read_csv("ratings.csv")

def manhattan_similarity(user_ratings1, user_ratings2):
    # Fusionnez les évaluations des deux utilisateurs
    merged_ratings = user_ratings1.join(user_ratings2, lsuffix="_user1", rsuffix="_user2").dropna()

    if len(merged_ratings) == 0:
        return 0  # Aucune évaluation commune, la similarité est nulle

    # Calculez la similarité de la distance de Manhattan
    similarity = 1 / (1 + cityblock(merged_ratings['rating_user1'], merged_ratings['rating_user2']))
    
    return similarity

def find_user_distance(userId1, userId2):
    user_ratings1 = find_user_ratings(userId1)
    user_ratings2 = find_user_ratings(userId2)
    
    similarity = manhattan_similarity(user_ratings1, user_ratings2)
    
    return [userId1, userId2, similarity]

def find_user_ratings(userID):
    user_ratings = ratings_dataframe.query(f"userId == {userID}")
    return user_ratings[["movieId", "rating"]].set_index("movieId")

def find_relative_distance(userId):
    users = ratings_dataframe["userId"].unique()
    users = users[users != userId]
    distances = [find_user_distance(userId, every_other_user_id) for every_other_user_id in users]
    
    return pd.DataFrame(distances, columns=["masterUserId", "userId", "similarity"])

def find_closest_users(userId, number_of_users):
    relative_distances = find_relative_distance(userId)
    relative_distances.sort_values("similarity", ascending=False, inplace=True)
    return relative_distances.head(number_of_users)

def make_recommendation(userId):
    user_ratings = find_user_ratings(userId)
    closest_users = find_closest_users(userId, 500)
    most_similar_user_id = closest_users.iloc[0]["userId"]
    
    closest_user_ratings = find_user_ratings(most_similar_user_id)
    unwatched_movies = closest_user_ratings.drop(user_ratings.index, errors="ignore")
    
    unwatched_movies = unwatched_movies.sort_values("rating", ascending=False)
    unwatched_movies = unwatched_movies.join(movies_dataframe)

    return unwatched_movies.head(10)

# Exemple d'utilisation
recommendations = make_recommendation(68)
print(recommendations)