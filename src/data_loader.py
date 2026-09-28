"""
Data Loader & Preprocessor for Movie Recommender System.
Loads MovieLens data, extracts metadata, builds mappings, and provides train/test splits.
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


class MovieDataLoader:
    """
    Manages loading, parsing, index-encoding, and sparse matrix generation
    for MovieLens datasets.
    """

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        self.data_dir = data_dir

        self.movies_df = None
        self.ratings_df = None
        self.tags_df = None

        # Mappings between raw IDs and 0-indexed integers
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.movie_to_idx = {}
        self.idx_to_movie = {}

        # Vocabulary of distinct genres
        self.unique_genres = []
        self.movie_genre_matrix = None  # shape: (num_movies, num_genres)

        # Train/Test splits
        self.train_ratings = None
        self.test_ratings = None

    def load_data(self):
        """Loads CSV files and initializes feature engineering."""
        movies_path = os.path.join(self.data_dir, "movies.csv")
        ratings_path = os.path.join(self.data_dir, "ratings.csv")
        tags_path = os.path.join(self.data_dir, "tags.csv")

        if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
            raise FileNotFoundError(f"Missing data files in {self.data_dir}. Run data/download_data.py first.")

        self.movies_df = pd.read_csv(movies_path)
        self.ratings_df = pd.read_csv(ratings_path)
        if os.path.exists(tags_path):
            self.tags_df = pd.read_csv(tags_path)
        else:
            self.tags_df = pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])

        self._preprocess_movies()
        self._build_index_mappings()
        self._build_genre_matrix()
        return self

    def _preprocess_movies(self):
        """Extracts release year, cleans titles, and joins tags for content representation."""
        # Extract 4-digit release year from title
        years = self.movies_df["title"].str.extract(r'\((\d{4})\)', expand=False)
        self.movies_df["year"] = pd.to_numeric(years, errors="coerce").fillna(0).astype(int)

        # Clean title without year
        self.movies_df["clean_title"] = (
            self.movies_df["title"]
            .str.replace(r'\s*\(\d{4}\)', '', regex=True)
            .str.strip()
        )

        # Aggregate tags per movie
        if not self.tags_df.empty:
            tag_agg = (
                self.tags_df.dropna(subset=["tag"])
                .groupby("movieId")["tag"]
                .apply(lambda tags: " ".join(tags.astype(str).unique()))
                .reset_index()
            )
            self.movies_df = self.movies_df.merge(tag_agg, on="movieId", how="left")
            self.movies_df["tag"] = self.movies_df["tag"].fillna("")
        else:
            self.movies_df["tag"] = ""

        # Format genres string (replace '|' with space for NLP tokenization)
        self.movies_df["genre_str"] = (
            self.movies_df["genres"]
            .fillna("")
            .str.replace("|", " ", regex=False)
            .str.replace("(no genres listed)", "", regex=False)
        )

        # Combined content string (genres + tags)
        self.movies_df["content_text"] = (
            self.movies_df["genre_str"] + " " + self.movies_df["tag"]
        ).str.strip()

    def _build_index_mappings(self):
        """Builds contiguous 0-indexed mappings for users and movies."""
        unique_users = sorted(self.ratings_df["userId"].unique())
        unique_movies = sorted(self.movies_df["movieId"].unique())

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for i, u in enumerate(unique_users)}

        self.movie_to_idx = {m: i for i, m in enumerate(unique_movies)}
        self.idx_to_movie = {i: m for i, m in enumerate(unique_movies)}

        # Add index columns into ratings_df
        # Only keep ratings for known movies in movies_df
        self.ratings_df = self.ratings_df[self.ratings_df["movieId"].isin(self.movie_to_idx)].copy()
        self.ratings_df["user_idx"] = self.ratings_df["userId"].map(self.user_to_idx)
        self.ratings_df["movie_idx"] = self.ratings_df["movieId"].map(self.movie_to_idx)

    def _build_genre_matrix(self):
        """Constructs a binary multi-hot genre matrix for all movies."""
        all_genres = set()
        for g_str in self.movies_df["genres"].dropna():
            for g in g_str.split("|"):
                if g and g != "(no genres listed)":
                    all_genres.add(g)
        self.unique_genres = sorted(list(all_genres))
        genre_to_idx = {g: i for i, g in enumerate(self.unique_genres)}

        matrix = np.zeros((len(self.movies_df), len(self.unique_genres)), dtype=np.float32)
        for _, row in self.movies_df.iterrows():
            m_idx = self.movie_to_idx.get(row["movieId"])
            if m_idx is not None and pd.notna(row["genres"]):
                for g in str(row["genres"]).split("|"):
                    if g in genre_to_idx:
                        matrix[m_idx, genre_to_idx[g]] = 1.0

        self.movie_genre_matrix = matrix

    def split_train_test(self, test_size=0.2, random_state=42):
        """Splits ratings into train and test DataFrames."""
        train_df, test_df = train_test_split(
            self.ratings_df,
            test_size=test_size,
            random_state=random_state,
            stratify=None
        )
        self.train_ratings = train_df.copy()
        self.test_ratings = test_df.copy()
        return self.train_ratings, self.test_ratings

    def build_sparse_matrix(self, ratings_df=None):
        """
        Builds a SciPy CSR sparse matrix of shape (num_users, num_movies).
        Unrated entries are 0.0.
        """
        if ratings_df is None:
            ratings_df = self.train_ratings if self.train_ratings is not None else self.ratings_df

        rows = ratings_df["user_idx"].values
        cols = ratings_df["movie_idx"].values
        data = ratings_df["rating"].values.astype(np.float32)

        num_users = len(self.user_to_idx)
        num_movies = len(self.movie_to_idx)

        sparse_mat = csr_matrix((data, (rows, cols)), shape=(num_users, num_movies), dtype=np.float32)
        return sparse_mat

    def get_sparsity_stats(self):
        """Calculates dataset dimensions and sparsity percentage."""
        n_users = len(self.user_to_idx)
        n_movies = len(self.movie_to_idx)
        n_ratings = len(self.ratings_df)
        total_possible = n_users * n_movies
        sparsity = (1.0 - (n_ratings / total_possible)) * 100.0

        return {
            "num_users": n_users,
            "num_movies": n_movies,
            "num_ratings": n_ratings,
            "sparsity_pct": round(sparsity, 2),
            "min_rating": float(self.ratings_df["rating"].min()),
            "max_rating": float(self.ratings_df["rating"].max()),
            "mean_rating": float(self.ratings_df["rating"].mean()),
        }

    def get_user_ratings(self, user_id):
        """Returns ratings history for a raw user_id."""
        sub = self.ratings_df[self.ratings_df["userId"] == user_id]
        return sub.merge(self.movies_df[["movieId", "title", "genres", "year"]], on="movieId")

    def get_movie_by_id(self, movie_id):
        """Returns metadata series for a raw movie_id."""
        match = self.movies_df[self.movies_df["movieId"] == movie_id]
        if not match.empty:
            return match.iloc[0]
        return None
