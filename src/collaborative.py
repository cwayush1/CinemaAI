"""
Collaborative Filtering Recommender using Scikit-Learn Matrix Factorization (TruncatedSVD)
and item-item collaborative similarity.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeRecommender:
    """
    Implements matrix factorization (TruncatedSVD) on centered user-item ratings
    and item-item collaborative filtering.
    """

    def __init__(self, data_loader, n_components=40, random_state=42):
        self.data_loader = data_loader
        self.n_components = n_components
        self.random_state = random_state

        self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self.user_means = None
        self.global_mean = 3.5
        self.reconstructed_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.train_sparse_mat = None

    def fit(self, train_ratings_df=None):
        """Fits SVD on centered user-item matrix."""
        if train_ratings_df is None:
            train_ratings_df = (
                self.data_loader.train_ratings
                if self.data_loader.train_ratings is not None
                else self.data_loader.ratings_df
            )

        self.global_mean = float(train_ratings_df["rating"].mean())
        num_users = len(self.data_loader.user_to_idx)
        num_movies = len(self.data_loader.movie_to_idx)

        # Compute per-user mean rating for centering
        user_sums = np.zeros(num_users, dtype=np.float64)
        user_counts = np.zeros(num_users, dtype=np.int64)

        for _, row in train_ratings_df.iterrows():
            u_idx = int(row["user_idx"])
            user_sums[u_idx] += row["rating"]
            user_counts[u_idx] += 1

        self.user_means = np.where(user_counts > 0, user_sums / np.maximum(user_counts, 1), self.global_mean)

        # Center ratings: (rating - user_mean)
        centered_ratings = train_ratings_df.copy()
        centered_ratings["centered_rating"] = (
            centered_ratings["rating"] - centered_ratings["user_idx"].map(lambda u: self.user_means[u])
        )

        rows = centered_ratings["user_idx"].values
        cols = centered_ratings["movie_idx"].values
        data = centered_ratings["centered_rating"].values.astype(np.float32)

        self.train_sparse_mat = csr_matrix((data, (rows, cols)), shape=(num_users, num_movies), dtype=np.float32)

        # Fit TruncatedSVD
        # U_reduced = svd.fit_transform(X), V = svd.components_
        self.user_factors = self.svd.fit_transform(self.train_sparse_mat)  # shape (num_users, n_components)
        self.item_factors = self.svd.components_  # shape (n_components, num_movies)

        # Precompute low-rank approximation for fast lookup
        # Shape: (num_users, num_movies)
        self.reconstructed_matrix = np.dot(self.user_factors, self.item_factors)
        return self

    def predict_rating(self, user_id, movie_id):
        """
        Predicts the rating given to movie_id by user_id.
        Falls back gracefully if user or movie is unseen.
        """
        u_idx = self.data_loader.user_to_idx.get(user_id)
        m_idx = self.data_loader.movie_to_idx.get(movie_id)

        if u_idx is None and m_idx is None:
            return self.global_mean
        if u_idx is None:
            # Fallback to item average if available
            return self.global_mean
        if m_idx is None:
            return float(self.user_means[u_idx])

        # Predicted rating = user_mean + SVD_reconstruction
        raw_pred = self.user_means[u_idx] + self.reconstructed_matrix[u_idx, m_idx]
        return float(np.clip(raw_pred, 0.5, 5.0))

    def predict_batch(self, user_indices, movie_indices):
        """Vectorized batch rating predictions for evaluation."""
        user_indices = np.asarray(user_indices)
        movie_indices = np.asarray(movie_indices)

        u_means = self.user_means[user_indices]
        # Dot product of corresponding user and item factor vectors
        reconstructed_vals = np.sum(
            self.user_factors[user_indices, :] * self.item_factors[:, movie_indices].T,
            axis=1
        )
        preds = u_means + reconstructed_vals
        return np.clip(preds, 0.5, 5.0)

    def recommend_for_user(self, user_id, top_n=10, exclude_rated=True, train_ratings_df=None):
        """Generates Top-N recommendations for user_id."""
        u_idx = self.data_loader.user_to_idx.get(user_id)
        if u_idx is None:
            return pd.DataFrame()

        if train_ratings_df is None:
            train_ratings_df = (
                self.data_loader.train_ratings
                if self.data_loader.train_ratings is not None
                else self.data_loader.ratings_df
            )

        rated_m_indices = set()
        if exclude_rated:
            user_ratings = train_ratings_df[train_ratings_df["userId"] == user_id]
            rated_m_indices = set(user_ratings["movie_idx"].values)

        user_mean = self.user_means[u_idx]
        predicted_ratings = user_mean + self.reconstructed_matrix[u_idx, :]

        # Mask already rated movies
        candidate_indices = [idx for idx in range(len(predicted_ratings)) if idx not in rated_m_indices]
        candidate_ratings = predicted_ratings[candidate_indices]

        top_candidates = np.argsort(candidate_ratings)[::-1][:top_n]

        results = []
        for rank_pos in top_candidates:
            m_idx = candidate_indices[rank_pos]
            raw_m_id = self.data_loader.idx_to_movie[m_idx]
            movie_info = self.data_loader.get_movie_by_id(raw_m_id)
            pred_score = float(np.clip(candidate_ratings[rank_pos], 0.5, 5.0))

            results.append({
                "movieId": raw_m_id,
                "title": movie_info["title"],
                "genres": movie_info["genres"],
                "year": movie_info["year"],
                "predicted_rating": round(pred_score, 3)
            })

        return pd.DataFrame(results)

    def get_similar_items(self, movie_id, top_n=10):
        """Finds items with similar latent representation vectors."""
        m_idx = self.data_loader.movie_to_idx.get(movie_id)
        if m_idx is None:
            return pd.DataFrame()

        target_vec = self.item_factors[:, m_idx].reshape(1, -1)
        sim_scores = cosine_similarity(target_vec, self.item_factors.T).flatten()
        sim_scores[m_idx] = -1.0

        top_indices = np.argsort(sim_scores)[::-1][:top_n]
        results = []
        for idx in top_indices:
            raw_m_id = self.data_loader.idx_to_movie[idx]
            movie_info = self.data_loader.get_movie_by_id(raw_m_id)
            results.append({
                "movieId": raw_m_id,
                "title": movie_info["title"],
                "genres": movie_info["genres"],
                "year": movie_info["year"],
                "latent_similarity": round(float(sim_scores[idx]), 4)
            })
        return pd.DataFrame(results)
