import pandas as pd
import numpy as np

# Chargez les données depuis les fichiers CSV
movies_dataframe = pd.read_csv("movies.csv")
ratings_dataframe = pd.read_csv("ratings.csv")

def pearson_similarity(user_ratings1, user_ratings2):
    # Rejoignez les évaluations des deux utilisateurs
    merged_ratings = user_ratings1.join(user_ratings2, lsuffix="_user1", rsuffix="_user2")
    
    if len(merged_ratings) == 0:
            return 0 
 
    # Calculez la moyenne des évaluations de chaque utilisateur
    mean_user1 = user_ratings1['rating'].mean()
    mean_user2 = user_ratings2['rating'].mean()

    # Calculez la similarité de Pearson
    numerator = ((merged_ratings['rating_user1'] - mean_user1) * (merged_ratings['rating_user2'] - mean_user2)).sum()
    denominator_user1 = np.sqrt(((merged_ratings['rating_user1'] - mean_user1) ** 2).sum())
    denominator_user2 = np.sqrt(((merged_ratings['rating_user2'] - mean_user2) ** 2).sum())
    denominator = (denominator_user1 * denominator_user2)

    if denominator == 0:
        return 0  # Évitez la division par zéro

    similarity = numerator / denominator
    return similarity

def find_user_distance(userId1, userId2):
    user_ratings1 = find_user_ratings(userId1)
    user_ratings2 = find_user_ratings(userId2)
    
    similarity = pearson_similarity(user_ratings1, user_ratings2)
    
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

def find_least_similar_user(userId):
    users = ratings_dataframe["userId"].unique()
    users = users[users != userId]
    
    min_similarity = float('inf')  # Initialisation de la valeur minimale de similarité à une valeur positive infinie
    least_similar_user = None  # Initialisation de l'utilisateur le moins similaire
    
    for user_id in users:
        similarity = pearson_similarity(find_user_ratings(userId), find_user_ratings(user_id))
        if similarity < min_similarity:
            min_similarity = similarity
            least_similar_user = user_id
    
    return least_similar_user, min_similarity

def make_recommendation(userId):
    user_ratings = find_user_ratings(userId)
    closest_users = find_closest_users(userId,10)
    most_similar_user_id = closest_users.iloc[0]["userId"]
    
    closest_user_ratings = find_user_ratings(most_similar_user_id)
    unwatched_movies = closest_user_ratings.drop(user_ratings.index, errors="ignore")
    
    unwatched_movies = unwatched_movies.sort_values("rating", ascending=False)
    #unwatched_movies = unwatched_movies.join(movies_dataframe)
    unwatched_movies = unwatched_movies.join(movies_dataframe.set_index('movieId'), on='movieId')
    return unwatched_movies.head(15)

# Exemple d'utilisation
#print(make_recommendation(9))   

#res = find_closest_users(81,15)
#print(res)

least_similar_user, min_similarity = find_least_similar_user(81)
print("L'utilisateur le moins similaire à l'utilisateur 81 est l'utilisateur", least_similar_user, "avec une similarité de", min_similarity)


'''
movies_dataframe.set_index("movieId", inplace = True)
total_rating_count = ratings_dataframe["movieId"].value_counts()

movies_dataframe["totalRatingCount"] = total_rating_count   #add total rating count column to movies dataframe

average_movie_ratings = ratings_dataframe.groupby("movieId").mean()["rating"]
movies_dataframe["averageRating"] = average_movie_ratings #add average rating column to movies dataframe

movies_dataframe.sort_values(["totalRatingCount", "averageRating"], ascending=False) #sort movies dataframe by total rating count
min_ratings_subset = movies_dataframe.query(f"totalRatingCount >= {100}") #query movies dataframe for movies with over 100 ratings

print(movies_dataframe.head())
'''