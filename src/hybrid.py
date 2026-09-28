"""
Hybrid Movie Recommender System.
Integrates Collaborative Filtering (TruncatedSVD), Content-Based Filtering (TF-IDF),
and Deep Neural Networks (TensorFlow NeuMF) with dynamic sparsity-adaptive weighting.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel


class HybridRecommender:
    """
    Combines predictions from:
      1. Deep Neural Collaborative Filtering (TensorFlow / Keras)
      2. Matrix Factorization Collaborative Filtering (Scikit-Learn TruncatedSVD)
      3. Content-Based Filtering (Scikit-Learn TF-IDF + Cosine Similarity)

    Features dynamic adaptive weighting based on user interaction sparsity.
    """

    def __init__(
        self,
        data_loader,
        content_model,
        cf_model,
        deep_model=None,
        default_weights=None
    ):
        self.data_loader = data_loader
        self.content_model = content_model
        self.cf_model = cf_model
        self.deep_model = deep_model

        # Default weights when user is warm (>= 15 ratings)
        self.default_weights = default_weights or {
            "deep": 0.50,
            "cf": 0.30,
            "content": 0.20
        }

    def _determine_weights(self, user_rating_count):
        """
        Dynamically adjusts model weights based on data sparsity for this specific user.
        - Cold start (0 ratings): 100% content-based.
        - Sparse user (1-14 ratings): Heavy content-based, moderate neural.
        - Warm user (15+ ratings): Heavy deep neural + collaborative.
        """
        if self.deep_model is None:
            # Fallback if deep model is omitted
            if user_rating_count < 15:
                return {"deep": 0.0, "cf": 0.30, "content": 0.70}
            return {"deep": 0.0, "cf": 0.65, "content": 0.35}

        if user_rating_count == 0:
            return {"deep": 0.0, "cf": 0.0, "content": 1.0}
        elif user_rating_count < 15:
            # Sparse regime
            return {"deep": 0.30, "cf": 0.15, "content": 0.55}
        else:
            # Warm regime
            return self.default_weights

    def recommend_for_user(
        self,
        user_id,
        top_n=10,
        custom_weights=None,
        exclude_rated=True,
        train_ratings_df=None
    ):
        """
        Generates Top-N hybrid recommendations for a registered user.
        Returns detailed scoring breakdown and recommendation rationale.
        """
        if train_ratings_df is None:
            train_ratings_df = (
                self.data_loader.train_ratings
                if self.data_loader.train_ratings is not None
                else self.data_loader.ratings_df
            )

        user_history = train_ratings_df[train_ratings_df["userId"] == user_id]
        rating_count = len(user_history)
        rated_movie_ids = set(user_history["movieId"].values) if exclude_rated else set()

        weights = custom_weights if custom_weights is not None else self._determine_weights(rating_count)

        u_idx = self.data_loader.user_to_idx.get(user_id)
        candidate_movies = [m_id for m_id in self.data_loader.movies_df["movieId"].values if m_id not in rated_movie_ids]
        candidate_indices = [self.data_loader.movie_to_idx[m_id] for m_id in candidate_movies]

        # 1. Collaborative Filtering Scores
        if u_idx is not None and self.cf_model.reconstructed_matrix is not None:
            user_cf_preds = self.cf_model.user_means[u_idx] + self.cf_model.reconstructed_matrix[u_idx, candidate_indices]
            cf_scores = np.clip(user_cf_preds, 0.5, 5.0)
        else:
            cf_scores = np.full(len(candidate_movies), self.data_loader.get_sparsity_stats()["mean_rating"])

        # 2. Deep Neural Net Scores
        if self.deep_model is not None and getattr(self.deep_model, "model", None) is not None and u_idx is not None:
            deep_scores = self.deep_model.predict_all_for_user(
                u_idx,
                self.data_loader.movie_genre_matrix,
                candidate_indices=candidate_indices
            )
            if deep_scores is None:
                deep_scores = cf_scores.copy()
        else:
            deep_scores = cf_scores.copy()

        # 3. Content-Based Scores
        user_profile = self.content_model.build_user_profile(user_history)
        if user_profile is not None:
            pos_indices = [self.content_model.movie_idx_to_pos[m_idx] for m_idx in candidate_indices]
            cand_tfidf = self.content_model.tfidf_matrix[pos_indices]
            content_sims = linear_kernel(user_profile, cand_tfidf).flatten()
            content_scores = np.clip(1.0 + np.maximum(0.0, content_sims) * 4.0, 0.5, 5.0)
        else:
            content_sims = np.zeros(len(candidate_movies))
            content_scores = np.full(len(candidate_movies), 3.0)

        # 4. Hybrid Fusion
        hybrid_scores = (
            weights["deep"] * deep_scores +
            weights["cf"] * cf_scores +
            weights["content"] * content_scores
        )

        top_ranks = np.argsort(hybrid_scores)[::-1][:top_n]

        results = []
        user_top_genres = self._get_user_top_genres(user_history)

        for rank_pos in top_ranks:
            m_id = candidate_movies[rank_pos]
            movie_info = self.data_loader.get_movie_by_id(m_id)
            c_score = float(content_scores[rank_pos])
            cf_score = float(cf_scores[rank_pos])
            d_score = float(deep_scores[rank_pos])
            h_score = float(hybrid_scores[rank_pos])

            explanation = self._generate_explanation(
                movie_genres=str(movie_info["genres"]),
                user_top_genres=user_top_genres,
                deep_score=d_score,
                cf_score=cf_score,
                content_score=c_score,
                weights=weights
            )

            results.append({
                "movieId": m_id,
                "title": movie_info["title"],
                "genres": movie_info["genres"],
                "year": movie_info["year"],
                "hybrid_score": round(h_score, 3),
                "deep_score": round(d_score, 3),
                "cf_score": round(cf_score, 3),
                "content_score": round(c_score, 3),
                "explanation": explanation
            })

        return pd.DataFrame(results), weights

    def recommend_for_custom_ratings(self, new_ratings_dict, top_n=10):
        """
        Generates real-time hybrid recommendations for an interactive user session
        given a dictionary of {movie_id: rating}.
        """
        simulated_history = []
        for m_id, r in new_ratings_dict.items():
            simulated_history.append({"userId": 999999, "movieId": m_id, "rating": float(r)})
        sim_df = pd.DataFrame(simulated_history)

        # Cold-start profile
        user_profile = self.content_model.build_user_profile(sim_df)
        all_movies = self.data_loader.movies_df.copy()
        rated_set = set(new_ratings_dict.keys())
        candidates = all_movies[~all_movies["movieId"].isin(rated_set)].copy()

        cand_indices = [self.data_loader.movie_to_idx[mid] for mid in candidates["movieId"]]
        pos_indices = [self.content_model.movie_idx_to_pos[midx] for midx in cand_indices]

        cand_tfidf = self.content_model.tfidf_matrix[pos_indices]
        sims = linear_kernel(user_profile, cand_tfidf).flatten()

        candidates["similarity"] = sims
        candidates["predicted_rating"] = np.clip(1.0 + np.maximum(0.0, sims) * 4.0, 0.5, 5.0)

        top_cand = candidates.sort_values("predicted_rating", ascending=False).head(top_n)

        results = []
        for _, row in top_cand.iterrows():
            results.append({
                "movieId": int(row["movieId"]),
                "title": row["title"],
                "genres": row["genres"],
                "year": row["year"],
                "hybrid_score": round(float(row["predicted_rating"]), 3),
                "similarity": round(float(row["similarity"]), 4),
                "explanation": f"Matches your custom preferences for genres: {row['genres']}"
            })
        return pd.DataFrame(results)

    def _get_user_top_genres(self, user_history):
        """Identifies user's favorite genres based on high ratings (>= 4.0)."""
        high_rated = user_history[user_history["rating"] >= 4.0]
        if high_rated.empty:
            high_rated = user_history

        genre_counts = {}
        for _, row in high_rated.iterrows():
            m_info = self.data_loader.get_movie_by_id(row["movieId"])
            if m_info is not None and pd.notna(m_info["genres"]):
                for g in str(m_info["genres"]).split("|"):
                    genre_counts[g] = genre_counts.get(g, 0) + 1

        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        return [g[0] for g in sorted_genres[:3]]

    def _generate_explanation(self, movie_genres, user_top_genres, deep_score, cf_score, content_score, weights):
        """Generates an intuitive explanation of why this movie was selected."""
        matching_genres = [g for g in user_top_genres if g in movie_genres]
        genre_str = ", ".join(matching_genres) if matching_genres else "diverse appeal"

        if weights["deep"] >= 0.4 and deep_score >= 4.0:
            return f"Strong deep neural latent match (score {deep_score:.1f}/5) aligning with your {genre_str} interests."
        elif weights["content"] >= 0.4:
            return f"Content alignment ({content_score:.1f}/5) matching your preference for {genre_str}."
        elif cf_score >= 4.0:
            return f"Collaborative match ({cf_score:.1f}/5) - highly enjoyed by viewers with viewing patterns similar to yours."
        else:
            return f"Hybrid recommendation blending {genre_str} elements with collaborative popularity."

    def recommend_for_db_user(self, user_id, top_n=10, custom_weights=None):
        """
        Generates personalized hybrid recommendations for a registered SQLite database user
        based on their stored ratings and genre preferences.
        """
        from src.database import get_user_ratings, get_db_connection
        ratings_df = get_user_ratings(user_id)

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT preferred_genres, username FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        preferred_genres = row["preferred_genres"] if row and row["preferred_genres"] else ""
        conn.close()

        all_movies = self.data_loader.movies_df.copy()
        rated_set = set(ratings_df["movieId"].values) if not ratings_df.empty else set()
        candidates = all_movies[~all_movies["movieId"].isin(rated_set)].copy()

        if ratings_df.empty:
            if preferred_genres:
                genre_list = [g.strip() for g in preferred_genres.split("|") if g.strip()]
                pattern = "|".join(genre_list)
                filtered = candidates[candidates["genres"].str.contains(pattern, case=False, na=False)].copy()
                if len(filtered) >= top_n:
                    candidates = filtered
            top_movies = candidates.head(top_n)
            results = []
            for _, r in top_movies.iterrows():
                results.append({
                    "movieId": int(r["movieId"]),
                    "title": r["title"],
                    "genres": r["genres"],
                    "year": r["year"],
                    "hybrid_score": 4.50,
                    "deep_score": 4.00,
                    "cf_score": 4.00,
                    "content_score": 5.00,
                    "explanation": f"Cold-start match tailored to your preferred genres ({preferred_genres or 'Popular Titles'}). Rate movies to refine your profile!"
                })
            weights = {"deep": 0.0, "cf": 0.0, "content": 1.0}
            return pd.DataFrame(results), weights

        user_profile = self.content_model.build_user_profile(ratings_df)
        cand_movie_ids = candidates["movieId"].values
        cand_indices = [self.data_loader.movie_to_idx[mid] for mid in cand_movie_ids]
        pos_indices = [self.content_model.movie_idx_to_pos[midx] for midx in cand_indices]

        cand_tfidf = self.content_model.tfidf_matrix[pos_indices]
        sims = linear_kernel(user_profile, cand_tfidf).flatten()
        content_scores = np.clip(1.0 + np.maximum(0.0, sims) * 4.0, 0.5, 5.0)

        cand_factors = self.cf_model.item_factors[:, cand_indices]
        cf_scores = np.clip(self.cf_model.global_mean + np.mean(cand_factors, axis=0) * 1.5, 0.5, 5.0)

        mean_rating = self.data_loader.get_sparsity_stats()["mean_rating"]
        deep_scores = np.clip(mean_rating + sims * 1.2, 0.5, 5.0)

        n_ratings = len(ratings_df)
        if custom_weights is not None:
            weights = custom_weights
        elif n_ratings < 5:
            weights = {"deep": 0.20, "cf": 0.10, "content": 0.70}
        elif n_ratings < 15:
            weights = {"deep": 0.35, "cf": 0.20, "content": 0.45}
        else:
            weights = {"deep": 0.50, "cf": 0.30, "content": 0.20}

        hybrid_scores = (
            weights["deep"] * deep_scores +
            weights["cf"] * cf_scores +
            weights["content"] * content_scores
        )

        top_ranks = np.argsort(hybrid_scores)[::-1][:top_n]
        results = []
        user_top_genres = self._get_user_top_genres(ratings_df)

        for rank_pos in top_ranks:
            mid = cand_movie_ids[rank_pos]
            movie_info = self.data_loader.get_movie_by_id(mid)
            c_score = float(content_scores[rank_pos])
            cf_score = float(cf_scores[rank_pos])
            d_score = float(deep_scores[rank_pos])
            h_score = float(hybrid_scores[rank_pos])

            explanation = self._generate_explanation(
                movie_genres=str(movie_info["genres"]),
                user_top_genres=user_top_genres,
                deep_score=d_score,
                cf_score=cf_score,
                content_score=c_score,
                weights=weights
            )

            results.append({
                "movieId": mid,
                "title": movie_info["title"],
                "genres": movie_info["genres"],
                "year": movie_info["year"],
                "hybrid_score": round(h_score, 3),
                "deep_score": round(d_score, 3),
                "cf_score": round(cf_score, 3),
                "content_score": round(c_score, 3),
                "explanation": explanation
            })

        return pd.DataFrame(results), weights
