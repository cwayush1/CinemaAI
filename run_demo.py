"""
CLI Demo for Movie Recommender System.
Quickly generate personalized recommendations or explore content similarity.
"""

import os
import sys
import argparse
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data_loader import MovieDataLoader
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.neural_cf import DeepRecommenderNet
from src.hybrid import HybridRecommender


def load_trained_system():
    saved_dir = os.path.join(BASE_DIR, "saved_models")
    artifacts_path = os.path.join(saved_dir, "artifacts.pkl")
    weights_path = os.path.join(saved_dir, "deep_recommender.weights.h5")

    if not os.path.exists(artifacts_path) or not os.path.exists(weights_path):
        print("[!] Pretrained models not found. Running training pipeline first...")
        from train import run_pipeline
        run_pipeline(epochs=5, batch_size=256)

    loader = MovieDataLoader()
    loader.load_data()
    loader.split_train_test()

    artifacts = joblib.load(artifacts_path)

    # Reconstruct Content Model
    content_model = ContentBasedRecommender(loader)
    content_model.vectorizer = artifacts["tfidf_vectorizer"]
    content_model.tfidf_matrix = artifacts["tfidf_matrix"]
    content_model.movie_idx_to_pos = artifacts["movie_idx_to_pos"]
    content_model.pos_to_movie_idx = artifacts["pos_to_movie_idx"]

    # Reconstruct CF Model
    cf_model = CollaborativeRecommender(loader, n_components=40)
    cf_model.user_means = artifacts["cf_user_means"]
    cf_model.user_factors = artifacts["cf_user_factors"]
    cf_model.item_factors = artifacts["cf_item_factors"]
    cf_model.reconstructed_matrix = artifacts["cf_reconstructed"]

    # Reconstruct Deep Neural Net
    num_users = len(loader.user_to_idx)
    num_movies = len(loader.movie_to_idx)
    genre_dim = loader.movie_genre_matrix.shape[1]

    deep_net = DeepRecommenderNet(num_users, num_movies, genre_dim, embedding_dim=32)
    deep_net.load(weights_path)

    # Hybrid Recommender
    hybrid = HybridRecommender(loader, content_model, cf_model, deep_net)
    return loader, content_model, cf_model, deep_net, hybrid


def main():
    parser = argparse.ArgumentParser(description="Movie Recommender CLI Demo")
    parser.add_argument("--user", type=int, default=1, help="Raw User ID to recommend for (e.g. 1)")
    parser.add_argument("--movie", type=int, default=None, help="Raw Movie ID for content similarity (e.g. 1 for Toy Story)")
    parser.add_argument("--top_n", type=int, default=5, help="Number of recommendations")
    args = parser.parse_args()

    loader, content_model, cf_model, deep_net, hybrid = load_trained_system()

    if args.movie is not None:
        movie_info = loader.get_movie_by_id(args.movie)
        if movie_info is None:
            print(f"[!] Movie ID {args.movie} not found.")
            return

        print("=" * 70)
        print(f"[*] CONTENT SIMILARITY FOR: {movie_info['title']} ({movie_info['genres']})")
        print("=" * 70)
        sim_df = content_model.get_similar_movies(args.movie, top_n=args.top_n)
        for i, row in sim_df.iterrows():
            print(f" {i+1}. {row['title']} | Score: {row['similarity_score']:.3f} | Genres: {row['genres']}")
    else:
        user_history = loader.get_user_ratings(args.user)
        print("=" * 70)
        print(f"[*] USER PROFILE: User ID {args.user} (Rated {len(user_history)} movies)")
        print("=" * 70)
        print("Top Rated Movies by this User:")
        top_user_movies = user_history.sort_values("rating", ascending=False).head(5)
        for _, row in top_user_movies.iterrows():
            print(f"  * {row['title']} (Rating: {row['rating']}) - {row['genres']}")

        print("\n" + "=" * 70)
        print(f"[*] TOP {args.top_n} HYBRID RECOMMENDATIONS FOR USER {args.user}")
        print("=" * 70)
        recs, weights = hybrid.recommend_for_user(args.user, top_n=args.top_n)
        print(f"Active Weights: Deep NN={weights['deep']:.2f}, SVD={weights['cf']:.2f}, Content={weights['content']:.2f}\n")

        for i, row in recs.iterrows():
            print(f"{i+1}. {row['title']} ({row['year']})")
            print(f"   Hybrid Score : {row['hybrid_score']:.2f}/5.0  (Deep: {row['deep_score']:.2f} | CF: {row['cf_score']:.2f} | Content: {row['content_score']:.2f})")
            print(f"   Genres       : {row['genres']}")
            print(f"   Why          : {row['explanation']}\n")


if __name__ == "__main__":
    main()
