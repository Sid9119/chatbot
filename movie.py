import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class MovieRecommender:
    def __init__(self, movies, ratings):
        self.movies = movies
        self.ratings = ratings
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.user_movie_matrix = None
        self.user_similarity_matrix = None

    def _compute_content_similarity(self):
        """Compute cosine similarity matrix for movie genres."""
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies['genres'])
        return cosine_similarity(tfidf_matrix, tfidf_matrix)

    def content_based_recommender(self, movie_title, n=3):
        """
        Recommend movies based on content similarity.
        
        Parameters:
        - movie_title: The title of the movie to base recommendations on.
        - n: Number of recommendations to return.
        
        Returns:
        - List of recommended movie titles.
        """
        cosine_sim = self._compute_content_similarity()
        idx = self.movies.index[self.movies['title'] == movie_title].tolist()[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.movies['title'].iloc[movie_indices]

    def _create_user_movie_matrix(self):
        """Create a user-item matrix and compute user similarities."""
        self.user_movie_matrix = self.ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
        self.user_similarity_matrix = cosine_similarity(self.user_movie_matrix)

    def collaborative_filtering_recommender(self, user_id, n=3):
        """
        Recommend movies based on collaborative filtering.
        
        Parameters:
        - user_id: The ID of the user for whom recommendations are being made.
        - n: Number of recommendations to return.
        
        Returns:
        - List of recommended movie titles.
        """
        if self.user_movie_matrix is None:
            self._create_user_movie_matrix()

        knn = NearestNeighbors(metric='cosine', algorithm='brute')
        knn.fit(self.user_movie_matrix)
        distances, indices = knn.kneighbors([self.user_movie_matrix.loc[user_id]], n_neighbors=n+1)
        neighbors_indices = indices.flatten()[1:]

        neighbor_ratings = self.user_movie_matrix.iloc[neighbors_indices].mean(axis=0)
        unseen_movies = self.user_movie_matrix.loc[user_id][self.user_movie_matrix.loc[user_id] == 0].index.tolist()
        recommendations = neighbor_ratings.loc[unseen_movies].sort_values(ascending=False).head(n)

        valid_recommendations = recommendations.index[recommendations.index.isin(self.movies['movie_id'])]

        if valid_recommendations.empty:
            return ["No recommendations available."]
        
        return self.movies.loc[self.movies['movie_id'].isin(valid_recommendations), 'title'].tolist()

# Example dataset
movies_df = pd.DataFrame({
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['Toy Story', 'Jumanji', 'Grumpier Old Men', 'Waiting to Exhale', 'Father of the Bride Part II'],
    'genres': ['Animation|Children|Comedy', 'Adventure|Children|Fantasy', 'Comedy|Romance', 'Comedy|Drama|Romance', 'Comedy']
})

ratings_df = pd.DataFrame({
    'user_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'movie_id': [1, 2, 3, 4, 5, 2, 3, 4, 5, 1],
    'rating': [5, 4, 3, 4, 5, 3, 4, 2, 5, 4]
})

# Initialize the recommender system
recommender = MovieRecommender(movies_df, ratings_df)

# Get recommendations
print("Content-Based Recommendations for 'Toy Story':")
print(recommender.content_based_recommender('Toy Story'))

print("\nCollaborative Filtering Recommendations for User 1:")
print(recommender.collaborative_filtering_recommender(1))
