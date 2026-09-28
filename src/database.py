"""
Database and Authentication Module for Movie Recommender System.
Provides persistent storage using SQLite for:
- User accounts (Sign Up, Sign In, Sign Out, Password Hashing with Salt)
- Personal movie ratings
- Personal watchlist / bookmarks
- User search and interaction activity logs
"""

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime
import pandas as pd

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "recommender.db"
)


def get_db_connection(db_path=DEFAULT_DB_PATH):
    """Returns a connection to the SQLite database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DEFAULT_DB_PATH):
    """Initializes tables for users, ratings, watchlist, and activity history."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    # Users Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            preferred_genres TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Personal Ratings Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            rating REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, movie_id),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Watchlist / Bookmarks Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, movie_id),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # User Activity History Log
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            activity_type TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    # Persistent User Sessions Table (persists login across page reloads/F5)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    conn.commit()

    # Check if a demo user exists, if not create one for instant testing
    cur.execute("SELECT id FROM users WHERE username = 'demo_user'")
    if not cur.fetchone():
        _create_demo_user(conn)

    conn.close()


def create_user_session(user_id, db_path=DEFAULT_DB_PATH):
    """Creates a persistent session token for user_id."""
    token = secrets.token_urlsafe(32)
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO user_sessions (session_token, user_id) VALUES (?, ?)",
        (token, user_id)
    )
    conn.commit()
    conn.close()
    return token


def get_user_by_session_token(token, db_path=DEFAULT_DB_PATH):
    """Retrieves user info using a session token. Returns user dict or None."""
    if not token:
        return None
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT u.id, u.username, u.email, u.preferred_genres
        FROM users u
        INNER JOIN user_sessions s ON u.id = s.user_id
        WHERE s.session_token = ?
    """, (token,))
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "preferred_genres": row["preferred_genres"]
        }
    return None


def delete_user_session(token, db_path=DEFAULT_DB_PATH):
    """Deletes a session token on sign out."""
    if not token:
        return
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM user_sessions WHERE session_token = ?", (token,))
    conn.commit()
    conn.close()


def _hash_password(password, salt=None):
    """PBKDF2 HMAC SHA-256 secure password hashing."""
    if salt is None:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return key.hex(), salt


def _create_demo_user(conn):
    """Creates a sample demo user with initial ratings and watchlist for convenience."""
    cur = conn.cursor()
    pwd_hash, salt = _hash_password("password123")
    cur.execute(
        "INSERT INTO users (username, email, password_hash, salt, preferred_genres) VALUES (?, ?, ?, ?, ?)",
        ("demo_user", "demo@example.com", pwd_hash, salt, "Sci-Fi|Action|Adventure")
    )
    user_id = cur.lastrowid

    # Seed demo ratings
    initial_ratings = [
        (user_id, 260, 5.0),   # Star Wars: Ep. IV
        (user_id, 1196, 5.0),  # Empire Strikes Back
        (user_id, 2571, 4.5),  # Matrix
        (user_id, 79132, 4.5), # Inception
        (user_id, 1, 4.0),     # Toy Story
        (user_id, 296, 4.0),   # Pulp Fiction
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO user_ratings (user_id, movie_id, rating) VALUES (?, ?, ?)",
        initial_ratings
    )

    # Seed demo watchlist
    initial_watchlist = [
        (user_id, 58559),  # The Dark Knight
        (user_id, 109487), # Interstellar
        (user_id, 318),    # Shawshank Redemption
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO watchlist (user_id, movie_id) VALUES (?, ?)",
        initial_watchlist
    )

    # Seed demo activity
    cur.execute(
        "INSERT INTO user_activity (user_id, activity_type, details) VALUES (?, ?, ?)",
        (user_id, "ACCOUNT_CREATED", "Welcome to Movie Recommender System!")
    )
    conn.commit()


def register_user(username, password, email=None, preferred_genres="", db_path=DEFAULT_DB_PATH):
    """Registers a new user account. Returns (True, user_dict) or (False, error_msg)."""
    username = username.strip()
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters long."
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long."

    pwd_hash, salt = _hash_password(password)

    conn = get_db_connection(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, email, password_hash, salt, preferred_genres) VALUES (?, ?, ?, ?, ?)",
            (username, email or "", pwd_hash, salt, preferred_genres)
        )
        user_id = cur.lastrowid
        cur.execute(
            "INSERT INTO user_activity (user_id, activity_type, details) VALUES (?, ?, ?)",
            (user_id, "SIGN_UP", f"Account created for {username}")
        )
        conn.commit()
        user = {
            "id": user_id,
            "username": username,
            "email": email or "",
            "preferred_genres": preferred_genres
        }
        return True, user
    except sqlite3.IntegrityError:
        return False, "Username is already taken. Please choose another."
    finally:
        conn.close()


def authenticate_user(username, password, db_path=DEFAULT_DB_PATH):
    """Verifies username and password. Returns (True, user_dict) or (False, error_msg)."""
    username = username.strip()
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT id, username, email, password_hash, salt, preferred_genres FROM users WHERE username = ?", (username,))
    row = cur.fetchone()

    if not row:
        conn.close()
        return False, "Username not found."

    user_id, u_name, email, stored_hash, salt, genres = row
    test_hash, _ = _hash_password(password, salt)

    if test_hash != stored_hash:
        conn.close()
        return False, "Incorrect password."

    # Log login activity
    cur.execute(
        "INSERT INTO user_activity (user_id, activity_type, details) VALUES (?, ?, ?)",
        (user_id, "SIGN_IN", "User signed in successfully")
    )
    conn.commit()
    conn.close()

    return True, {
        "id": user_id,
        "username": u_name,
        "email": email,
        "preferred_genres": genres
    }


def save_user_rating(user_id, movie_id, rating, movie_title="", db_path=DEFAULT_DB_PATH):
    """Saves or updates a rating given by the logged-in user."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO user_ratings (user_id, movie_id, rating, timestamp)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id, movie_id) DO UPDATE SET
            rating = excluded.rating,
            timestamp = CURRENT_TIMESTAMP
    """, (user_id, movie_id, float(rating)))

    cur.execute(
        "INSERT INTO user_activity (user_id, activity_type, details) VALUES (?, ?, ?)",
        (user_id, "RATED_MOVIE", f"Rated '{movie_title}' with {rating} stars")
    )
    conn.commit()
    conn.close()


def delete_user_rating(user_id, movie_id, db_path=DEFAULT_DB_PATH):
    """Deletes a rating given by the user."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM user_ratings WHERE user_id = ? AND movie_id = ?", (user_id, movie_id))
    cur.execute(
        "INSERT INTO user_activity (user_id, activity_type, details) VALUES (?, ?, ?)",
        (user_id, "DELETED_RATING", f"Removed rating for movie ID {movie_id}")
    )
    conn.commit()
    conn.close()


def get_user_ratings(user_id, db_path=DEFAULT_DB_PATH):
    """Retrieves all personal ratings for a user as a DataFrame."""
    conn = get_db_connection(db_path)
    df = pd.read_sql_query(
        "SELECT movie_id as movieId, rating, timestamp FROM user_ratings WHERE user_id = ? ORDER BY timestamp DESC",
        conn,
        params=(user_id,)
    )
    conn.close()
    return df


def add_to_watchlist(user_id, movie_id, movie_title="", db_path=DEFAULT_DB_PATH):
    """Adds a movie to the user's personal watchlist."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT OR IGNORE INTO watchlist (user_id, movie_id) VALUES (?, ?)",
            (user_id, movie_id)
        )
        cur.execute(
            "INSERT INTO user_activity (user_id, activity_type, details) VALUES (?, ?, ?)",
            (user_id, "WATCHLIST_ADD", f"Added '{movie_title}' to watchlist")
        )
        conn.commit()
    finally:
        conn.close()


def remove_from_watchlist(user_id, movie_id, db_path=DEFAULT_DB_PATH):
    """Removes a movie from the user's personal watchlist."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM watchlist WHERE user_id = ? AND movie_id = ?", (user_id, movie_id))
    cur.execute(
        "INSERT INTO user_activity (user_id, activity_type, details) VALUES (?, ?, ?)",
        (user_id, "WATCHLIST_REMOVE", f"Removed movie ID {movie_id} from watchlist")
    )
    conn.commit()
    conn.close()


def get_user_watchlist(user_id, db_path=DEFAULT_DB_PATH):
    """Retrieves all movie IDs currently in user's watchlist."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT movie_id, added_at FROM watchlist WHERE user_id = ? ORDER BY added_at DESC", (user_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"movie_id": r["movie_id"], "added_at": r["added_at"]} for r in rows]


def is_in_watchlist(user_id, movie_id, db_path=DEFAULT_DB_PATH):
    """Returns True if movie is in user's watchlist."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM watchlist WHERE user_id = ? AND movie_id = ?", (user_id, movie_id))
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def get_user_activity(user_id, limit=20, db_path=DEFAULT_DB_PATH):
    """Fetches user activity logs."""
    conn = get_db_connection(db_path)
    df = pd.read_sql_query(
        "SELECT activity_type, details, timestamp FROM user_activity WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
        conn,
        params=(user_id, limit)
    )
    conn.close()
    return df


def get_user_profile_stats(user_id, db_path=DEFAULT_DB_PATH):
    """Computes summary statistics for a user profile."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*), AVG(rating) FROM user_ratings WHERE user_id = ?", (user_id,))
    rating_count, avg_rating = cur.fetchone()

    cur.execute("SELECT COUNT(*) FROM watchlist WHERE user_id = ?", (user_id,))
    watchlist_count = cur.fetchone()[0]

    conn.close()
    return {
        "rating_count": rating_count or 0,
        "avg_rating": round(avg_rating, 2) if avg_rating is not None else 0.0,
        "watchlist_count": watchlist_count or 0
    }
