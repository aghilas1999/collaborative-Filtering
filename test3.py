import pandas as pd
import numpy as np
from scipy.spatial.distance import correlation

# Charger les données des évaluations des utilisateurs
ratings = pd.read_csv('ratings.csv')  # Assurez-vous de remplacer par le chemin réel du fichier de données des évaluations

# Charger les données des films pour obtenir les titres
movies = pd.read_csv('movies.csv')  # Assurez-vous de remplacer par le chemin réel du fichier de données des films

# Définir l'utilisateur cible pour lequel vous souhaitez faire une recommandation
user_id_cible = 1  # Remplacez par l'ID de l'utilisateur cible

# Calculer la similarité de Pearson entre l'utilisateur cible et tous les autres utilisateurs
similarities = []
user_ratings_cible = ratings[ratings['userId'] == user_id_cible]
for user_id in ratings['userId'].unique():
    if user_id != user_id_cible:
        user_ratings = ratings[ratings['userId'] == user_id]

        # Trouver les films communs entre les deux utilisateurs
        common_movies = np.intersect1d(user_ratings_cible['movieId'], user_ratings['movieId'])

        if len(common_movies) > 0:
            u = user_ratings_cible[user_ratings_cible['movieId'].isin(common_movies)]['rating']
            v = user_ratings[user_ratings['movieId'].isin(common_movies)]['rating']

            # Ajouter une petite valeur pour éviter la division par zéro
            epsilon = 1e-9
            similarity = 1 - correlation(u, v) + epsilon
            similarities.append((user_id, similarity))

# Trier les utilisateurs similaires par ordre décroissant de similarité
similarities.sort(key=lambda x: x[1], reverse=True)


# Sélectionner les 10 utilisateurs les plus similaires (par exemple)
top_similar_users = similarities[:2]


# Trouver les films aimés par les utilisateurs similaires mais non vus par l'utilisateur cible
movies_seen_by_user_cible = ratings[ratings['userId'] == user_id_cible]['movieId'].tolist()
recommended_movies = set()

for user_id, _ in top_similar_users:
    user_ratings = ratings[ratings['userId'] == user_id]
    recommended_movies.update(user_ratings[user_ratings['rating'] >= 4]['movieId'].tolist())

recommended_movies = list(recommended_movies - set(movies_seen_by_user_cible))

# Classer les films recommandés en fonction de leur popularité (par exemple, le nombre d'avis positifs)
movie_counts = ratings[ratings['movieId'].isin(recommended_movies)]['movieId'].value_counts()
recommended_movies = sorted(recommended_movies, key=lambda x: -movie_counts.get(x, 0))

# Afficher les films recommandés
# Afficher les dix meilleurs films recommandés
print("Top 10 des films recommandés pour l'utilisateur", user_id_cible)
for movie_id in recommended_movies[:10]:
    movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
    print(movie_title)