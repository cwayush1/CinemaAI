"""
End-to-End Model Training Pipeline.
Trains Content-Based, Collaborative Filtering (TruncatedSVD), and Deep Neural Network (NeuMF),
evaluates metrics on test split, and persists trained artifacts.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Ensure utf-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data_loader import MovieDataLoader
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.neural_cf import DeepRecommenderNet
from src.hybrid import HybridRecommender
from src.evaluate import Evaluator


def run_pipeline(epochs=6, batch_size=256, embedding_dim=32):
    print("=" * 70)
    print("[*] MOVIE RECOMMENDER SYSTEM: FULL PIPELINE TRAINING")
    print("=" * 70)

    # 1. Load Data
    print("\n[Step 1/6] Loading & Preprocessing MovieLens Dataset...")
    loader = MovieDataLoader()
    loader.load_data()
    stats = loader.get_sparsity_stats()
    print(f"  • Unique Users    : {stats['num_users']:,}")
    print(f"  • Unique Movies   : {stats['num_movies']:,}")
    print(f"  • Total Ratings   : {stats['num_ratings']:,}")
    print(f"  • Matrix Sparsity : {stats['sparsity_pct']}%")
    print(f"  • Rating Range    : {stats['min_rating']} - {stats['max_rating']} (Mean: {stats['mean_rating']:.2f})")

    # 2. Train/Test Split
    print("\n[Step 2/6] Splitting Dataset into Train (80%) and Test (20%)...")
    train_df, test_df = loader.split_train_test(test_size=0.2, random_state=42)
    print(f"  • Train Samples: {len(train_df):,}")
    print(f"  • Test Samples : {len(test_df):,}")

    # 3. Fit Content-Based Model
    print("\n[Step 3/6] Training Content-Based Filtering (TF-IDF + Cosine Sim)...")
    content_model = ContentBasedRecommender(loader)
    content_model.fit()
    print(f"  • Content TF-IDF Vocabulary Size: {content_model.tfidf_matrix.shape[1]:,} features")

    # 4. Fit Collaborative Filtering (TruncatedSVD)
    print("\n[Step 4/6] Training Collaborative Filtering (TruncatedSVD Matrix Factorization)...")
    cf_model = CollaborativeRecommender(loader, n_components=40)
    cf_model.fit(train_df)
    print(f"  • SVD Latent Components: {cf_model.n_components}")
    print(f"  • User Factors Matrix  : {cf_model.user_factors.shape}")
    print(f"  • Item Factors Matrix  : {cf_model.item_factors.shape}")

    # 5. Train Deep Neural Network (NeuMF)
    print("\n[Step 5/6] Training Deep Neural Collaborative Network (TensorFlow/Keras)...")
    num_users = len(loader.user_to_idx)
    num_movies = len(loader.movie_to_idx)
    genre_dim = loader.movie_genre_matrix.shape[1]

    deep_net = DeepRecommenderNet(
        num_users=num_users,
        num_movies=num_movies,
        genre_dim=genre_dim,
        embedding_dim=embedding_dim,
        mlp_units=(128, 64, 32),
        dropout_rate=0.2,
        l2_reg=1e-5
    )

    deep_net.train(
        train_df=train_df,
        val_df=test_df,
        genre_matrix=loader.movie_genre_matrix,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # 6. Evaluation on Test Set
    print("\n[Step 6/6] Evaluating Models on Test Set...")
    evaluator = Evaluator(loader, test_df)

    cf_metrics = evaluator.evaluate_rating_prediction(cf_model, model_type="cf")
    deep_metrics = evaluator.evaluate_rating_prediction(deep_net, model_type="deep")

    # Hybrid recommender instance
    hybrid_recommender = HybridRecommender(loader, content_model, cf_model, deep_net)

    print("\nComputing Ranking Metrics (Precision@10, Recall@10, NDCG@10)...")
    cf_ranking = evaluator.evaluate_ranking(lambda u, top_n: cf_model.recommend_for_user(u, top_n=top_n), k=10)
    hybrid_ranking = evaluator.evaluate_ranking(
        lambda u, top_n: hybrid_recommender.recommend_for_user(u, top_n=top_n)[0], k=10
    )

    # Sparsity Analysis
    print("Evaluating Performance Across User Sparsity Tiers...")
    sparsity_df = evaluator.evaluate_sparsity_impact(cf_model, deep_net)

    print("\n" + "=" * 70)
    print("📊 BENCHMARK EVALUATION RESULTS")
    print("=" * 70)
    print(f"Model: Collaborative Filtering (TruncatedSVD)")
    print(f"  • RMSE: {cf_metrics['rmse']} | MAE: {cf_metrics['mae']}")
    print(f"  • Precision@10: {cf_ranking['precision@10']} | Recall@10: {cf_ranking['recall@10']} | NDCG@10: {cf_ranking['ndcg@10']}")
    print("-" * 70)
    print(f"Model: Deep Neural Network (TensorFlow NeuMF + Content)")
    print(f"  • RMSE: {deep_metrics['rmse']} | MAE: {deep_metrics['mae']}")
    print("-" * 70)
    print(f"Model: Hybrid Recommender (Deep + CF + Content)")
    print(f"  • Precision@10: {hybrid_ranking['precision@10']} | Recall@10: {hybrid_ranking['recall@10']} | NDCG@10: {hybrid_ranking['ndcg@10']}")
    print("-" * 70)
    print("Sparsity Impact Comparison:")
    print(sparsity_df.to_string(index=False))
    print("=" * 70)

    # Persist Models & Artifacts
    saved_dir = os.path.join(BASE_DIR, "saved_models")
    os.makedirs(saved_dir, exist_ok=True)

    weights_path = os.path.join(saved_dir, "deep_recommender.weights.h5")
    deep_net.save(weights_path)
    print(f"\n[Saved] Deep Neural Network weights saved to: {weights_path}")

    artifacts_path = os.path.join(saved_dir, "artifacts.pkl")
    artifacts = {
        "user_to_idx": loader.user_to_idx,
        "idx_to_user": loader.idx_to_user,
        "movie_to_idx": loader.movie_to_idx,
        "idx_to_movie": loader.idx_to_movie,
        "unique_genres": loader.unique_genres,
        "cf_user_means": cf_model.user_means,
        "cf_user_factors": cf_model.user_factors,
        "cf_item_factors": cf_model.item_factors,
        "cf_reconstructed": cf_model.reconstructed_matrix,
        "tfidf_vectorizer": content_model.vectorizer,
        "tfidf_matrix": content_model.tfidf_matrix,
        "movie_idx_to_pos": content_model.movie_idx_to_pos,
        "pos_to_movie_idx": content_model.pos_to_movie_idx,
    }
    joblib.dump(artifacts, artifacts_path)
    print(f"[Saved] Precomputed factors & vectorizers saved to: {artifacts_path}")

    summary = {
        "stats": stats,
        "cf_metrics": cf_metrics,
        "cf_ranking": cf_ranking,
        "deep_metrics": deep_metrics,
        "hybrid_ranking": hybrid_ranking,
        "sparsity_results": sparsity_df.to_dict(orient="records")
    }
    with open(os.path.join(saved_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Saved] Evaluation summary JSON saved to: {os.path.join(saved_dir, 'evaluation_summary.json')}")
    print("\n[SUCCESS] Training and artifact persistence complete!")
    return loader, content_model, cf_model, deep_net, hybrid_recommender


if __name__ == "__main__":
    run_pipeline(epochs=6, batch_size=256, embedding_dim=32)
