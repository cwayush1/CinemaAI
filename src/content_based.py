"""
Content-Based Filtering Recommender using Scikit-Learn TF-IDF and Cosine Similarity.
Builds item feature representations from genres & tags and constructs user preference profiles.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBasedRecommender:
    """
    Computes movie-movie similarities and personalized user-movie scores
    leveraging TF-IDF vectors of genres and descriptive tags.
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_matrix = None
        self.movie_idx_to_pos = {}
        self.pos_to_movie_idx = {}

    def fit(self):
        """Fits TF-IDF vectorizer on movie content strings."""
        movies_df = self.data_loader.movies_df.copy()
        # Sort consistently by movie_idx
        movies_df["movie_idx"] = movies_df["movieId"].map(self.data_loader.movie_to_idx)
        movies_df = movies_df.dropna(subset=["movie_idx"]).sort_values("movie_idx").reset_index(drop=True)

        for pos, row in movies_df.iterrows():
            m_idx = int(row["movie_idx"])
            self.movie_idx_to_pos[m_idx] = pos
            self.pos_to_movie_idx[pos] = m_idx

        content_corpus = movies_df["content_text"].fillna("").values
        self.tfidf_matrix = self.vectorizer.fit_transform(content_corpus)
        return self

    def get_similar_movies(self, movie_id, top_n=10):
        """
        Finds the top N most similar movies to a given movie_id based on content.
        Returns a DataFrame with movie metadata and similarity scores.
        """
        m_idx = self.data_loader.movie_to_idx.get(movie_id)
        if m_idx is None or m_idx not in self.movie_idx_to_pos:
            return pd.DataFrame()

        pos = self.movie_idx_to_pos[m_idx]
        target_vec = self.tfidf_matrix[pos]

        # Cosine similarity via linear kernel on normalized TF-IDF vectors
        sim_scores = linear_kernel(target_vec, self.tfidf_matrix).flatten()

        # Exclude the movie itself
        sim_scores[pos] = -1.0

        top_indices = np.argsort(sim_scores)[::-1][:top_n]

        results = []
        for idx in top_indices:
            orig_m_idx = self.pos_to_movie_idx[idx]
            raw_m_id = self.data_loader.idx_to_movie[orig_m_idx]
            movie_info = self.data_loader.get_movie_by_id(raw_m_id)
            results.append({
                "movieId": raw_m_id,
                "title": movie_info["title"],
                "genres": movie_info["genres"],
                "year": movie_info["year"],
                "similarity_score": round(float(sim_scores[idx]), 4)
            })

        return pd.DataFrame(results)

    def build_user_profile(self, user_ratings_df):
        """
        Constructs a weighted user preference vector across TF-IDF features.
        Positive weights for ratings > mean, negative for ratings < mean.
        """
        if user_ratings_df.empty:
            return None

        mean_rating = user_ratings_df["rating"].mean()
        profile_vec = np.zeros((1, self.tfidf_matrix.shape[1]), dtype=np.float32)
        total_weight = 0.0

        for _, row in user_ratings_df.iterrows():
            m_id = row["movieId"]
            m_idx = self.data_loader.movie_to_idx.get(m_id)
            if m_idx is None or m_idx not in self.movie_idx_to_pos:
                continue

            pos = self.movie_idx_to_pos[m_idx]
            # Center rating: e.g. 5.0 - 3.5 = +1.5, 1.0 - 3.5 = -2.5
            weight = row["rating"] - mean_rating
            item_vec = self.tfidf_matrix[pos].toarray()
            profile_vec += weight * item_vec
            total_weight += abs(weight)

        if total_weight > 0:
            profile_vec /= total_weight

        # L2 normalize user profile vector
        norm = np.linalg.norm(profile_vec)
        if norm > 0:
            profile_vec /= norm

        return profile_vec

    def predict_for_user(self, user_id, train_ratings_df=None, top_n=10, exclude_rated=True):
        """
        Scores all candidate movies for a user by comparing their preference profile
        with item content vectors.
        """
        if train_ratings_df is None:
            train_ratings_df = self.data_loader.ratings_df

        user_history = train_ratings_df[train_ratings_df["userId"] == user_id]
        if user_history.empty:
            # Cold-start fallback: return top rated/popular movies
            return pd.DataFrame()

        user_profile = self.build_user_profile(user_history)
        if user_profile is None:
            return pd.DataFrame()

        sim_scores = linear_kernel(user_profile, self.tfidf_matrix).flatten()

        rated_movie_ids = set(user_history["movieId"].unique()) if exclude_rated else set()

        scored_movies = []
        for pos, score in enumerate(sim_scores):
            m_idx = self.pos_to_movie_idx[pos]
            raw_m_id = self.data_loader.idx_to_movie[m_idx]
            if raw_m_id in rated_movie_ids:
                continue

            # Scale cosine similarity [-1, 1] to rating range [1.0, 5.0]
            # typical cosine score is [0, 1]
            predicted_rating = 1.0 + (max(0.0, float(score)) * 4.0)

            movie_info = self.data_loader.get_movie_by_id(raw_m_id)
            scored_movies.append({
                "movieId": raw_m_id,
                "title": movie_info["title"],
                "genres": movie_info["genres"],
                "year": movie_info["year"],
                "content_score": round(float(score), 4),
                "predicted_rating": round(float(predicted_rating), 3)
            })

        scored_df = pd.DataFrame(scored_movies).sort_values("content_score", ascending=False)
        return scored_df.head(top_n).reset_index(drop=True)

    def predict_single(self, user_profile, movie_id):
        """Predicts score for a specific movie given a precomputed user profile."""
        m_idx = self.data_loader.movie_to_idx.get(movie_id)
        if m_idx is None or m_idx not in self.movie_idx_to_pos or user_profile is None:
            return 3.0
        pos = self.movie_idx_to_pos[m_idx]
        sim = float(linear_kernel(user_profile, self.tfidf_matrix[pos])[0][0])
        # Scale to 1.0 - 5.0
        return float(np.clip(1.0 + max(0.0, sim) * 4.0, 0.5, 5.0))
