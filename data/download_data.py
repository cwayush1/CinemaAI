"""
Data Downloader & Preprocessor for Movie Recommender System.
Fetches the MovieLens (ml-latest-small) dataset or synthesizes a high-fidelity
fallback dataset if offline.
"""

import os
import sys
import io
import zipfile
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def create_sample_dataset():
    """Generates a realistic synthetic MovieLens dataset if internet is unavailable."""
    print("Creating synthetic MovieLens-compatible dataset...")
    movies_data = [
        (1, "Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy"),
        (2, "Jumanji (1995)", "Adventure|Children|Fantasy"),
        (3, "Grumpier Old Men (1995)", "Comedy|Romance"),
        (4, "Waiting to Exhale (1995)", "Comedy|Drama|Romance"),
        (5, "Father of the Bride Part II (1995)", "Comedy"),
        (6, "Heat (1995)", "Action|Crime|Thriller"),
        (7, "Sabrina (1995)", "Comedy|Romance"),
        (8, "Tom and Huck (1995)", "Adventure|Children"),
        (9, "Sudden Death (1995)", "Action"),
        (10, "GoldenEye (1995)", "Action|Adventure|Thriller"),
        (11, "American President, The (1995)", "Comedy|Drama|Romance"),
        (12, "Dracula: Dead and Loving It (1995)", "Comedy|Horror"),
        (13, "Balto (1995)", "Adventure|Animation|Children"),
        (14, "Nixon (1995)", "Drama"),
        (15, "Cutthroat Island (1995)", "Action|Adventure|Romance"),
        (16, "Casino (1995)", "Crime|Drama"),
        (17, "Sense and Sensibility (1995)", "Drama|Romance"),
        (18, "Four Rooms (1995)", "Comedy"),
        (19, "Ace Ventura: When Nature Calls (1995)", "Comedy"),
        (20, "Money Train (1995)", "Action|Comedy|Crime|Drama|Thriller"),
        (21, "Get Shorty (1995)", "Comedy|Crime|Thriller"),
        (22, "Copycat (1995)", "Crime|Drama|Horror|Mystery|Thriller"),
        (23, "Assassins (1995)", "Action|Crime|Thriller"),
        (24, "Powder (1995)", "Drama|Sci-Fi"),
        (25, "Leaving Las Vegas (1995)", "Drama|Romance"),
        (26, "Othello (1995)", "Drama"),
        (27, "Now and Then (1995)", "Children|Drama"),
        (28, "Persuasion (1995)", "Drama|Romance"),
        (29, "City of Lost Children, The (1995)", "Adventure|Drama|Fantasy|Mystery|Sci-Fi"),
        (30, "Shanghai Triad (1995)", "Crime|Drama"),
        (31, "Dangerous Minds (1995)", "Drama"),
        (32, "Twelve Monkeys (1995)", "Mystery|Sci-Fi|Thriller"),
        (34, "Babe (1995)", "Children|Drama"),
        (36, "Dead Man Walking (1995)", "Crime|Drama"),
        (39, "Clueless (1995)", "Comedy|Romance"),
        (47, "Seven (a.k.a. Se7en) (1995)", "Mystery|Thriller"),
        (48, "Pocahontas (1995)", "Animation|Children|Drama|Musical|Romance"),
        (50, "Usual Suspects, The (1995)", "Crime|Mystery|Thriller"),
        (110, "Braveheart (1995)", "Action|Drama|War"),
        (150, "Apollo 13 (1995)", "Adventure|Drama|IMAX"),
        (260, "Star Wars: Episode IV - A New Hope (1977)", "Action|Adventure|Sci-Fi"),
        (296, "Pulp Fiction (1994)", "Comedy|Crime|Drama|Thriller"),
        (318, "Shawshank Redemption, The (1994)", "Crime|Drama"),
        (356, "Forrest Gump (1994)", "Comedy|Drama|Romance|War"),
        (480, "Jurassic Park (1993)", "Action|Adventure|Sci-Fi|Thriller"),
        (527, "Schindler's List (1993)", "Drama|War"),
        (589, "Terminator 2: Judgment Day (1991)", "Action|Sci-Fi"),
        (593, "Silence of the Lambs, The (1991)", "Crime|Horror|Thriller"),
        (858, "Godfather, The (1972)", "Crime|Drama"),
        (1196, "Star Wars: Episode V - The Empire Strikes Back (1980)", "Action|Adventure|Sci-Fi"),
        (1198, "Raiders of the Lost Ark (1981)", "Action|Adventure"),
        (1210, "Star Wars: Episode VI - Return of the Jedi (1983)", "Action|Adventure|Sci-Fi"),
        (1270, "Back to the Future (1985)", "Adventure|Comedy|Sci-Fi"),
        (2571, "Matrix, The (1999)", "Action|Sci-Fi|Thriller"),
        (2959, "Fight Club (1999)", "Action|Crime|Drama|Thriller"),
        (4993, "Lord of the Rings: The Fellowship of the Ring, The (2001)", "Adventure|Fantasy"),
        (5952, "Lord of the Rings: The Two Towers, The (2002)", "Adventure|Fantasy"),
        (7153, "Lord of the Rings: The Return of the King, The (2003)", "Action|Adventure|Drama|Fantasy"),
        (58559, "Dark Knight, The (2008)", "Action|Crime|Drama|IMAX"),
        (68954, "Up (2009)", "Adventure|Animation|Children|Comedy"),
        (79132, "Inception (2010)", "Action|Crime|Drama|Mystery|Sci-Fi|Thriller|IMAX"),
        (88125, "Harry Potter and the Deathly Hallows: Part 2 (2011)", "Action|Adventure|Drama|Fantasy|Mystery|IMAX"),
        (109487, "Interstellar (2014)", "Sci-Fi|IMAX"),
        (112852, "Whiplash (2014)", "Drama"),
        (122904, "Deadpool (2016)", "Action|Adventure|Comedy|Sci-Fi"),
        (134853, "Inside Out (2015)", "Adventure|Animation|Children|Comedy|Drama|Fantasy"),
        (168252, "Logan (2017)", "Action|Sci-Fi"),
        (177765, "Coco (2017)", "Adventure|Animation|Children|Fantasy"),
        (187596, "Avengers: Infinity War (2018)", "Action|Adventure|Sci-Fi"),
        (193587, "Spider-Man: Into the Spider-Verse (2018)", "Action|Adventure|Animation|Sci-Fi"),
    ]

    movies_df = pd.DataFrame(movies_data, columns=["movieId", "title", "genres"])
    movies_df.to_csv(os.path.join(DATA_DIR, "movies.csv"), index=False)

    np.random.seed(42)
    users = list(range(1, 101))  # 100 users
    movie_ids = movies_df["movieId"].values

    ratings_list = []
    # Generate ratings with natural preference clusters
    for u in users:
        # User preference bias towards certain genre keywords
        pref_sci_fi = np.random.rand() > 0.5
        pref_crime = np.random.rand() > 0.5
        n_ratings = np.random.randint(20, 60)
        chosen_movies = np.random.choice(movie_ids, size=n_ratings, replace=False)

        for m_id in chosen_movies:
            movie_genres = movies_df.loc[movies_df["movieId"] == m_id, "genres"].values[0]
            base_rating = 3.5 + np.random.normal(0, 0.7)
            if pref_sci_fi and "Sci-Fi" in movie_genres:
                base_rating += 0.8
            if pref_crime and "Crime" in movie_genres:
                base_rating += 0.7

            rating = np.clip(np.round(base_rating * 2) / 2, 0.5, 5.0)
            timestamp = 1600000000 + np.random.randint(0, 50000000)
            ratings_list.append((u, m_id, rating, timestamp))

    ratings_df = pd.DataFrame(ratings_list, columns=["userId", "movieId", "rating", "timestamp"])
    ratings_df.to_csv(os.path.join(DATA_DIR, "ratings.csv"), index=False)

    tags_data = [
        (1, 260, "classic sci-fi", 1600001000),
        (2, 296, "cult classic", 1600002000),
        (3, 318, "masterpiece", 1600003000),
        (4, 58559, "batman", 1600004000),
        (5, 79132, "mind-bending", 1600005000),
        (6, 109487, "space exploration", 1600006000),
        (7, 2571, "cyberpunk", 1600007000),
        (8, 2959, "psychological twist", 1600008000),
    ]
    tags_df = pd.DataFrame(tags_data, columns=["userId", "movieId", "tag", "timestamp"])
    tags_df.to_csv(os.path.join(DATA_DIR, "tags.csv"), index=False)

    print(f"Synthetic dataset saved: {len(movies_df)} movies, {len(ratings_df)} ratings, {len(tags_df)} tags.")


def download_movielens():
    """Attempts to download the official MovieLens ml-latest-small dataset."""
    movies_file = os.path.join(DATA_DIR, "movies.csv")
    ratings_file = os.path.join(DATA_DIR, "ratings.csv")

    print("Attempting to download MovieLens dataset from GroupLens...")
    try:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(
            MOVIELENS_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            zip_bytes = resp.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                for filename in z.namelist():
                    basename = os.path.basename(filename)
                    if basename in ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]:
                        with open(os.path.join(DATA_DIR, basename), "wb") as f_out:
                            f_out.write(z.read(filename))
        print("MovieLens ml-latest-small dataset downloaded and extracted successfully!")
    except Exception as e:
        print(f"Network download failed ({e}). Falling back to synthetic MovieLens dataset.")
        if not (os.path.exists(movies_file) and os.path.exists(ratings_file)):
            create_sample_dataset()


if __name__ == "__main__":
    download_movielens()
