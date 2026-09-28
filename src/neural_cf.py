"""
Deep Neural Network Recommender System using TensorFlow / Keras.
Implements Neural Collaborative Filtering (NeuMF) with Content Embedding Injection,
Batch Normalization, Dropout, and L2 Regularization to handle sparse user-item interaction data.
"""

import os
import numpy as np
import pandas as pd
try:
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers, Model, optimizers
    HAS_TF = True
except (ImportError, ModuleNotFoundError):
    HAS_TF = False


class DeepRecommenderNet:
    """
    Two-branch Neural Collaborative Filtering network:
    1. Generalized Matrix Factorization (GMF) branch learning linear latent interactions.
    2. Deep Multi-Layer Perceptron (MLP) branch learning non-linear feature interactions
       incorporating item content (genre multi-hot vectors).
    """

    def __init__(
        self,
        num_users,
        num_movies,
        genre_dim,
        embedding_dim=64,
        mlp_units=(128, 64, 32),
        dropout_rate=0.2,
        l2_reg=1e-5,
        min_rating=0.5,
        max_rating=5.0
    ):
        self.num_users = num_users
        self.num_movies = num_movies
        self.genre_dim = genre_dim
        self.embedding_dim = embedding_dim
        self.mlp_units = mlp_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.min_rating = min_rating
        self.max_rating = max_rating

        if HAS_TF:
            self.model = self._build_model()
        else:
            self.model = None
        self.history = None

    def _build_model(self):
        """Constructs the Keras functional model."""
        # --- Inputs ---
        user_input = layers.Input(shape=(1,), dtype="int32", name="user_input")
        movie_input = layers.Input(shape=(1,), dtype="int32", name="movie_input")
        genre_input = layers.Input(shape=(self.genre_dim,), dtype="float32", name="genre_input")

        # --- GMF (Generalized Matrix Factorization) Embeddings ---
        gmf_user_embed = layers.Embedding(
            input_dim=self.num_users,
            output_dim=self.embedding_dim,
            embeddings_regularizer=regularizers.l2(self.l2_reg),
            name="gmf_user_embed"
        )(user_input)
        gmf_movie_embed = layers.Embedding(
            input_dim=self.num_movies,
            output_dim=self.embedding_dim,
            embeddings_regularizer=regularizers.l2(self.l2_reg),
            name="gmf_movie_embed"
        )(movie_input)

        gmf_user_vec = layers.Flatten(name="gmf_user_flat")(gmf_user_embed)
        gmf_movie_vec = layers.Flatten(name="gmf_movie_flat")(gmf_movie_embed)

        # GMF Interaction (Element-wise dot product)
        gmf_vector = layers.Multiply(name="gmf_product")([gmf_user_vec, gmf_movie_vec])

        # --- MLP Embeddings & Content Representation ---
        mlp_user_embed = layers.Embedding(
            input_dim=self.num_users,
            output_dim=self.embedding_dim,
            embeddings_regularizer=regularizers.l2(self.l2_reg),
            name="mlp_user_embed"
        )(user_input)
        mlp_movie_embed = layers.Embedding(
            input_dim=self.num_movies,
            output_dim=self.embedding_dim,
            embeddings_regularizer=regularizers.l2(self.l2_reg),
            name="mlp_movie_embed"
        )(movie_input)

        mlp_user_vec = layers.Flatten(name="mlp_user_flat")(mlp_user_embed)
        mlp_movie_vec = layers.Flatten(name="mlp_movie_flat")(mlp_movie_embed)

        # Content projection dense layer
        genre_dense = layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name="genre_dense"
        )(genre_input)

        # Concatenate for deep representation
        mlp_vector = layers.Concatenate(name="mlp_concat")([mlp_user_vec, mlp_movie_vec, genre_dense])

        # Deep MLP layers with Batch Normalization & Dropout to prevent overfitting on sparse data
        for i, units in enumerate(self.mlp_units):
            mlp_vector = layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name=f"mlp_dense_{i+1}"
            )(mlp_vector)
            mlp_vector = layers.BatchNormalization(name=f"mlp_bn_{i+1}")(mlp_vector)
            mlp_vector = layers.Dropout(self.dropout_rate, name=f"mlp_dropout_{i+1}")(mlp_vector)

        # User and Item Bias Embeddings
        user_bias = layers.Embedding(input_dim=self.num_users, output_dim=1, name="user_bias")(user_input)
        movie_bias = layers.Embedding(input_dim=self.num_movies, output_dim=1, name="movie_bias")(movie_input)
        user_bias_vec = layers.Flatten(name="user_bias_flat")(user_bias)
        movie_bias_vec = layers.Flatten(name="movie_bias_flat")(movie_bias)

        # --- NeuMF Fusion ---
        fusion = layers.Concatenate(name="fusion_concat")(
            [gmf_vector, mlp_vector, user_bias_vec, movie_bias_vec]
        )

        output = layers.Dense(1, activation="linear", name="rating_output")(fusion)

        model = Model(
            inputs=[user_input, movie_input, genre_input],
            outputs=output,
            name="Deep_Neural_Recommender"
        )

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss="mean_squared_error",
            metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")]
        )

        return model

    def train(
        self,
        train_df,
        val_df,
        genre_matrix,
        epochs=15,
        batch_size=256,
        verbose=1
    ):
        """Trains the deep neural network with validation monitoring and callbacks."""
        X_train_user = train_df["user_idx"].values
        X_train_movie = train_df["movie_idx"].values
        X_train_genre = genre_matrix[X_train_movie]
        y_train = train_df["rating"].values.astype(np.float32)

        X_val_user = val_df["user_idx"].values
        X_val_movie = val_df["movie_idx"].values
        X_val_genre = genre_matrix[X_val_movie]
        y_val = val_df["rating"].values.astype(np.float32)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-5,
                verbose=1
            )
        ]

        self.history = self.model.fit(
            x={
                "user_input": X_train_user,
                "movie_input": X_train_movie,
                "genre_input": X_train_genre
            },
            y=y_train,
            validation_data=(
                {
                    "user_input": X_val_user,
                    "movie_input": X_val_movie,
                    "genre_input": X_val_genre
                },
                y_val
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return self.history

    def predict_batch(self, user_indices, movie_indices, genre_matrix):
        """Vectorized predictions for evaluation."""
        if not HAS_TF or self.model is None:
            return np.full(len(user_indices), (self.min_rating + self.max_rating) / 2.0)

        user_indices = np.asarray(user_indices, dtype=np.int32)
        movie_indices = np.asarray(movie_indices, dtype=np.int32)
        genres = genre_matrix[movie_indices]

        preds = self.model.predict(
            {
                "user_input": user_indices,
                "movie_input": movie_indices,
                "genre_input": genres
            },
            batch_size=1024,
            verbose=0
        ).flatten()

        return np.clip(preds, self.min_rating, self.max_rating)

    def predict_all_for_user(self, user_idx, genre_matrix, candidate_indices=None):
        """Fast prediction for all movies for a given user."""
        if not HAS_TF or self.model is None:
            return None

        if candidate_indices is None:
            candidate_indices = np.arange(self.num_movies, dtype=np.int32)
        else:
            candidate_indices = np.asarray(candidate_indices, dtype=np.int32)

        user_arr = np.full(len(candidate_indices), user_idx, dtype=np.int32)
        genre_arr = genre_matrix[candidate_indices]

        preds = self.model.predict(
            {
                "user_input": user_arr,
                "movie_input": candidate_indices,
                "genre_input": genre_arr
            },
            batch_size=2048,
            verbose=0
        ).flatten()

        return np.clip(preds, self.min_rating, self.max_rating)

    def extract_movie_embeddings(self):
        """Extracts learned movie embedding representations for analysis and visualization."""
        if not HAS_TF or self.model is None:
            return None

        gmf_layer = self.model.get_layer("gmf_movie_embed")
        mlp_layer = self.model.get_layer("mlp_movie_embed")

        gmf_weights = gmf_layer.get_weights()[0]  # shape (num_movies, embedding_dim)
        mlp_weights = mlp_layer.get_weights()[0]  # shape (num_movies, embedding_dim)

        combined = np.hstack([gmf_weights, mlp_weights])
        return combined

    def save(self, filepath):
        """Saves model weights."""
        if HAS_TF and self.model is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save_weights(filepath)

    def load(self, filepath):
        """Loads model weights."""
        if HAS_TF and self.model is not None:
            self.model.load_weights(filepath)
