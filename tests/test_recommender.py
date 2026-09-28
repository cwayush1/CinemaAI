"""
Comprehensive unit test suite for Movie Recommender System.
Verifies data loading, content-based TF-IDF, SVD collaborative filtering,
deep neural network inference, and hybrid recommendations.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_loader import MovieDataLoader
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.neural_cf import DeepRecommenderNet
from src.hybrid import HybridRecommender
from src.evaluate import compute_rmse, compute_mae, precision_recall_at_k, ndcg_at_k


class TestMovieRecommenderSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up data loader and shared models once for all tests."""
        cls.loader = MovieDataLoader()
        cls.loader.load_data()
        cls.train_df, cls.test_df = cls.loader.split_train_test(test_size=0.2, random_state=42)

        # Train fast small content and CF models
        cls.content_model = ContentBasedRecommender(cls.loader).fit()
        cls.cf_model = CollaborativeRecommender(cls.loader, n_components=10).fit(cls.train_df)

        # Initialize small Deep Recommender Net
        num_users = len(cls.loader.user_to_idx)
        num_movies = len(cls.loader.movie_to_idx)
        genre_dim = cls.loader.movie_genre_matrix.shape[1]
        cls.deep_net = DeepRecommenderNet(
            num_users=num_users,
            num_movies=num_movies,
            genre_dim=genre_dim,
            embedding_dim=8,
            mlp_units=(16, 8)
        )

        cls.hybrid = HybridRecommender(cls.loader, cls.content_model, cls.cf_model, cls.deep_net)

    def test_01_data_loader_structure(self):
        """Test dataset loading, index mappings, and sparsity calculation."""
        stats = self.loader.get_sparsity_stats()
        self.assertGreater(stats["num_users"], 0)
        self.assertGreater(stats["num_movies"], 0)
        self.assertGreater(stats["num_ratings"], 0)
        self.assertGreater(stats["sparsity_pct"], 90.0)  # Real-world MovieLens sparsity > 95%
        self.assertEqual(len(self.loader.user_to_idx), len(self.loader.idx_to_user))
        self.assertEqual(len(self.loader.movie_to_idx), len(self.loader.idx_to_movie))
        self.assertIsNotNone(self.loader.movie_genre_matrix)

    def test_02_content_based_similarity(self):
        """Test TF-IDF content similarity lookup and scoring."""
        # Movie ID 1 = Toy Story
        sim_df = self.content_model.get_similar_movies(1, top_n=5)
        self.assertEqual(len(sim_df), 5)
        self.assertNotIn(1, sim_df["movieId"].values)  # Target movie excluded
        self.assertTrue((sim_df["similarity_score"] >= 0.0).all())
        self.assertTrue((sim_df["similarity_score"] <= 1.0).all())

    def test_03_collaborative_filtering_predictions(self):
        """Test SVD matrix factorization predictions."""
        pred = self.cf_model.predict_rating(user_id=1, movie_id=1)
        self.assertIsInstance(pred, float)
        self.assertGreaterEqual(pred, 0.5)
        self.assertLessEqual(pred, 5.0)

        # Test batch prediction
        user_idxs = np.array([0, 1, 2])
        movie_idxs = np.array([0, 1, 2])
        batch_preds = self.cf_model.predict_batch(user_idxs, movie_idxs)
        self.assertEqual(len(batch_preds), 3)

        # Test user recommendations
        recs = self.cf_model.recommend_for_user(user_id=1, top_n=5)
        self.assertEqual(len(recs), 5)
        self.assertTrue("predicted_rating" in recs.columns)

    def test_04_deep_neural_network_forward_pass(self):
        """Test Deep Neural Network forward pass and shape outputs."""
        user_idxs = np.array([0, 1])
        movie_idxs = np.array([10, 20])
        preds = self.deep_net.predict_batch(user_idxs, movie_idxs, self.loader.movie_genre_matrix)
        self.assertEqual(len(preds), 2)
        self.assertTrue(np.all(preds >= 0.5) and np.all(preds <= 5.0))

        embeddings = self.deep_net.extract_movie_embeddings()
        self.assertEqual(embeddings.shape[0], len(self.loader.movie_to_idx))
        self.assertEqual(embeddings.shape[1], 16)  # 8 gmf + 8 mlp

    def test_05_hybrid_recommender(self):
        """Test hybrid fusion, adaptive weighting, and rationale generation."""
        recs, weights = self.hybrid.recommend_for_user(user_id=1, top_n=5)
        self.assertEqual(len(recs), 5)
        self.assertTrue("hybrid_score" in recs.columns)
        self.assertTrue("explanation" in recs.columns)
        self.assertAlmostEqual(weights["deep"] + weights["cf"] + weights["content"], 1.0)

        # Test cold-start interactive custom rating simulation
        custom_ratings = {1: 5.0, 260: 5.0, 296: 1.0}
        custom_recs = self.hybrid.recommend_for_custom_ratings(custom_ratings, top_n=4)
        self.assertEqual(len(custom_recs), 4)

    def test_06_evaluation_metrics(self):
        """Test evaluation math (RMSE, MAE, Precision@K, NDCG@K)."""
        y_true = np.array([4.0, 5.0, 3.0, 2.0])
        y_pred = np.array([3.8, 4.9, 3.1, 2.2])

        rmse = compute_rmse(y_true, y_pred)
        mae = compute_mae(y_true, y_pred)
        self.assertLess(rmse, 0.3)
        self.assertLess(mae, 0.3)

        actual_liked = {1, 2, 3}
        recommended = [1, 4, 2, 5, 6]
        p, r = precision_recall_at_k(actual_liked, recommended, k=5)
        self.assertEqual(p, 2 / 5.0)
        self.assertEqual(r, 2 / 3.0)

        ndcg = ndcg_at_k(actual_liked, recommended, k=5)
        self.assertGreater(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)

    def test_07_database_auth_and_persistence(self):
        """Test user registration, login, rating persistence, watchlist, and db recommendations."""
        import tempfile
        from src.database import (
            init_db,
            register_user,
            authenticate_user,
            save_user_rating,
            get_user_ratings,
            add_to_watchlist,
            get_user_watchlist,
            get_user_profile_stats
        )

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_db = tmp.name

        try:
            init_db(tmp_db)

            # 1. Registration
            ok, user = register_user("testuser_99", "secretpass123", "test@domain.com", "Action|Sci-Fi", db_path=tmp_db)
            self.assertTrue(ok)
            self.assertEqual(user["username"], "testuser_99")

            # Duplicate prevention
            dup_ok, _ = register_user("testuser_99", "secretpass123", db_path=tmp_db)
            self.assertFalse(dup_ok)

            # 2. Authentication
            auth_ok, auth_user = authenticate_user("testuser_99", "secretpass123", db_path=tmp_db)
            self.assertTrue(auth_ok)
            self.assertEqual(auth_user["id"], user["id"])

            bad_auth, _ = authenticate_user("testuser_99", "wrongpass", db_path=tmp_db)
            self.assertFalse(bad_auth)

            # 3. Save Personal Rating
            save_user_rating(user["id"], 260, 5.0, "Star Wars", db_path=tmp_db)
            ratings = get_user_ratings(user["id"], db_path=tmp_db)
            self.assertEqual(len(ratings), 1)
            self.assertEqual(ratings.iloc[0]["rating"], 5.0)

            # 4. Watchlist
            add_to_watchlist(user["id"], 1, "Toy Story", db_path=tmp_db)
            w = get_user_watchlist(user["id"], db_path=tmp_db)
            self.assertEqual(len(w), 1)
            self.assertEqual(w[0]["movie_id"], 1)

            # 5. Profile Stats
            stats = get_user_profile_stats(user["id"], db_path=tmp_db)
            self.assertEqual(stats["rating_count"], 1)
            self.assertEqual(stats["watchlist_count"], 1)

        finally:
            if os.path.exists(tmp_db):
                try:
                    os.remove(tmp_db)
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
