"""
Evaluation Suite for Recommender Systems.
Computes rating prediction metrics (RMSE, MAE) and ranking metrics (Precision@K, Recall@K, NDCG@K),
plus sparsity bucket stress-testing.
"""

import numpy as np
import pandas as pd


def compute_rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def compute_mae(y_true, y_pred):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def precision_recall_at_k(actual_items, recommended_items, k=10):
    """
    Computes Precision@K and Recall@K.
    actual_items: set of items user actually liked (e.g. rating >= 4.0 in test set)
    recommended_items: ordered list of top recommendations
    """
    if not actual_items or not recommended_items:
        return 0.0, 0.0

    top_k = recommended_items[:k]
    hits = len(set(top_k).intersection(set(actual_items)))
    precision = hits / float(k)
    recall = hits / float(len(actual_items))
    return precision, recall


def ndcg_at_k(actual_items, recommended_items, k=10):
    """Normalized Discounted Cumulative Gain at K."""
    if not actual_items or not recommended_items:
        return 0.0

    top_k = recommended_items[:k]
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in actual_items:
            dcg += 1.0 / np.log2((i + 1) + 1.0)

    # Ideal DCG
    idcg = sum(1.0 / np.log2((i + 1) + 1.0) for i in range(min(len(actual_items), k)))
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


class Evaluator:
    """Evaluates recommender algorithms on test partitions."""

    def __init__(self, data_loader, test_df=None):
        self.data_loader = data_loader
        self.test_df = test_df if test_df is not None else data_loader.test_ratings

    def evaluate_rating_prediction(self, model, model_type="cf"):
        """
        Evaluates RMSE and MAE on the test split.
        model_type: 'cf', 'deep', or 'content'
        """
        test_df = self.test_df.copy()
        y_true = test_df["rating"].values

        if model_type == "cf":
            preds = model.predict_batch(test_df["user_idx"].values, test_df["movie_idx"].values)
        elif model_type == "deep":
            preds = model.predict_batch(
                test_df["user_idx"].values,
                test_df["movie_idx"].values,
                self.data_loader.movie_genre_matrix
            )
        elif model_type == "content":
            # For each user, predict using their content profile
            preds = []
            user_groups = self.data_loader.train_ratings.groupby("userId")
            profile_cache = {}

            for _, row in test_df.iterrows():
                u_id = row["userId"]
                m_id = row["movieId"]
                if u_id not in profile_cache:
                    u_hist = user_groups.get_group(u_id) if u_id in user_groups.groups else pd.DataFrame()
                    profile_cache[u_id] = model.build_user_profile(u_hist)

                pred = model.predict_single(profile_cache[u_id], m_id)
                preds.append(pred)
            preds = np.array(preds)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        rmse_val = compute_rmse(y_true, preds)
        mae_val = compute_mae(y_true, preds)

        return {"rmse": round(rmse_val, 4), "mae": round(mae_val, 4)}

    def evaluate_ranking(self, recommender_fn, sample_users=50, k=10, threshold=4.0):
        """
        Evaluates Precision@K, Recall@K, and NDCG@K over a sample of test users.
        """
        test_df = self.test_df[self.test_df["rating"] >= threshold]
        active_users = test_df["userId"].unique()

        np.random.seed(42)
        if len(active_users) > sample_users:
            eval_users = np.random.choice(active_users, size=sample_users, replace=False)
        else:
            eval_users = active_users

        precisions = []
        recalls = []
        ndcgs = []

        for u_id in eval_users:
            actual_liked = set(test_df[test_df["userId"] == u_id]["movieId"].values)
            if not actual_liked:
                continue

            rec_df = recommender_fn(u_id, top_n=k)
            if isinstance(rec_df, tuple):
                rec_df = rec_df[0]

            if rec_df.empty:
                continue

            recommended_ids = rec_df["movieId"].tolist()

            p, r = precision_recall_at_k(actual_liked, recommended_ids, k=k)
            n = ndcg_at_k(actual_liked, recommended_ids, k=k)

            precisions.append(p)
            recalls.append(r)
            ndcgs.append(n)

        return {
            f"precision@{k}": round(float(np.mean(precisions)), 4) if precisions else 0.0,
            f"recall@{k}": round(float(np.mean(recalls)), 4) if recalls else 0.0,
            f"ndcg@{k}": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0
        }

    def evaluate_sparsity_impact(self, cf_model, deep_model):
        """
        Analyzes performance across user interaction sparsity buckets:
          - Sparse: < 30 ratings
          - Moderate: 30 - 100 ratings
          - Dense: > 100 ratings
        Shows how Deep Neural Network with Content Embeddings outperforms pure CF on sparse users.
        """
        user_counts = self.data_loader.train_ratings["userId"].value_counts()
        test_with_counts = self.test_df.copy()
        test_with_counts["train_count"] = test_with_counts["userId"].map(user_counts).fillna(0)

        buckets = {
            "Sparse (<30 ratings)": test_with_counts[test_with_counts["train_count"] < 30],
            "Moderate (30-100 ratings)": test_with_counts[
                (test_with_counts["train_count"] >= 30) & (test_with_counts["train_count"] <= 100)
            ],
            "Dense (>100 ratings)": test_with_counts[test_with_counts["train_count"] > 100],
        }

        results = []
        for bucket_name, sub_df in buckets.items():
            if sub_df.empty:
                continue

            y_true = sub_df["rating"].values

            # CF predictions
            cf_preds = cf_model.predict_batch(sub_df["user_idx"].values, sub_df["movie_idx"].values)
            cf_rmse = compute_rmse(y_true, cf_preds)

            # Deep predictions
            deep_preds = deep_model.predict_batch(
                sub_df["user_idx"].values,
                sub_df["movie_idx"].values,
                self.data_loader.movie_genre_matrix
            )
            deep_rmse = compute_rmse(y_true, deep_preds)

            results.append({
                "sparsity_bucket": bucket_name,
                "test_samples": len(sub_df),
                "cf_rmse": round(cf_rmse, 4),
                "deep_rmse": round(deep_rmse, 4),
                "deep_improvement_pct": round(((cf_rmse - deep_rmse) / cf_rmse) * 100.0, 2)
            })

        return pd.DataFrame(results)
