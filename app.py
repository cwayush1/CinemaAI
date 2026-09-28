"""
Full-Stack Interactive Streamlit Application for CinemaAI Movie Recommender.
Features:
- Dedicated Full-Page Sign In / Register Portal (No sidebar cramping)
- Decluttered, Cinema-First User Experience
- Real-Time Hybrid Recommendations with Theatrical Movie Posters
- Personal Watchlist & Ratings Library
- Separate Technical AI & Benchmark Explorer
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import linear_kernel

# Set Page Configuration
st.set_page_config(
    page_title="CinemaAI | Movie Recommender",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.database import (
    init_db,
    register_user,
    authenticate_user,
    save_user_rating,
    delete_user_rating,
    get_user_ratings,
    add_to_watchlist,
    remove_from_watchlist,
    get_user_watchlist,
    is_in_watchlist,
    get_user_activity,
    get_user_profile_stats
)
from src.data_loader import MovieDataLoader
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.neural_cf import DeepRecommenderNet
from src.hybrid import HybridRecommender
from src.poster_fetcher import get_movie_poster

# Initialize database
init_db()


@st.cache_resource(show_spinner=False)
def load_all_models_and_data():
    """Loads dataset and trained models into memory once."""
    loader = MovieDataLoader()
    loader.load_data()
    loader.split_train_test()

    saved_dir = os.path.join(BASE_DIR, "saved_models")
    artifacts_path = os.path.join(saved_dir, "artifacts.pkl")
    weights_path = os.path.join(saved_dir, "deep_recommender.weights.h5")
    summary_path = os.path.join(saved_dir, "evaluation_summary.json")

    if not os.path.exists(artifacts_path) or not os.path.exists(weights_path):
        from train import run_pipeline
        run_pipeline(epochs=5, batch_size=256)

    artifacts = joblib.load(artifacts_path)

    # Content Model
    content_model = ContentBasedRecommender(loader)
    content_model.vectorizer = artifacts["tfidf_vectorizer"]
    content_model.tfidf_matrix = artifacts["tfidf_matrix"]
    content_model.movie_idx_to_pos = artifacts["movie_idx_to_pos"]
    content_model.pos_to_movie_idx = artifacts["pos_to_movie_idx"]

    # Collaborative SVD Model
    cf_model = CollaborativeRecommender(loader, n_components=40)
    cf_model.user_means = artifacts["cf_user_means"]
    cf_model.user_factors = artifacts["cf_user_factors"]
    cf_model.item_factors = artifacts["cf_item_factors"]
    cf_model.reconstructed_matrix = artifacts["cf_reconstructed"]

    # Deep Neural Network (NeuMF)
    num_users = len(loader.user_to_idx)
    num_movies = len(loader.movie_to_idx)
    genre_dim = loader.movie_genre_matrix.shape[1]

    deep_net = DeepRecommenderNet(num_users, num_movies, genre_dim, embedding_dim=32)
    deep_net.load(weights_path)

    hybrid = HybridRecommender(loader, content_model, cf_model, deep_net)

    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

    return loader, content_model, cf_model, deep_net, hybrid, summary


loader, content_model, cf_model, deep_net, hybrid, summary = load_all_models_and_data()

# Session State for User Authentication
if "user" not in st.session_state:
    st.session_state["user"] = None

if "guest_mode" not in st.session_state:
    st.session_state["guest_mode"] = False

# Sleek Custom Styling
st.markdown("""
<style>
    /* Card aesthetics */
    .movie-card {
        background-color: #151921;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        border: 1px solid #242c38;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .movie-card:hover {
        border-color: #3b82f6;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        background-color: #1e293b;
        color: #93c5fd;
    }
    .badge-match {
        background-color: #064e3b;
        color: #a7f3d0;
        font-weight: bold;
        font-size: 12px;
        padding: 4px 10px;
        border-radius: 8px;
    }
    .auth-container {
        background-color: #151921;
        border-radius: 16px;
        padding: 36px;
        border: 1px solid #2d3748;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        margin-top: 20px;
    }
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0 16px 0;
        border-bottom: 1px solid #242c38;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# VIEW 1: DEDICATED AUTHENTICATION PAGE (Sign In / Register)
# =========================================================================
if st.session_state["user"] is None and not st.session_state["guest_mode"]:
    _, col_auth, _ = st.columns([1, 2, 1])

    with col_auth:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="font-size: 42px; margin-bottom: 6px;">🍿 CinemaAI</h1>
            <p style="color: #94a3b8; font-size: 16px;">
                Discover movies you'll love with personalized hybrid AI recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        auth_tab_sign_in, auth_tab_register = st.tabs(["🔑 Sign In", "📝 Create Account"])

        # --- SIGN IN TAB ---
        with auth_tab_sign_in:
            st.markdown("##### Welcome Back")
            si_username = st.text_input("Username", key="auth_si_user", placeholder="Enter your username")
            si_password = st.text_input("Password", type="password", key="auth_si_pwd", placeholder="Enter your password")

            c_btn1, c_btn2 = st.columns([1, 1])
            with c_btn1:
                if st.button("Sign In 🚀", type="primary", use_container_width=True):
                    if not si_username or not si_password:
                        st.error("Please enter both username and password.")
                    else:
                        ok, res = authenticate_user(si_username, si_password)
                        if ok:
                            st.session_state["user"] = res
                            st.success(f"Welcome back, {res['username']}!")
                            st.rerun()
                        else:
                            st.error(res)

            with c_btn2:
                if st.button("⚡ Instant Demo Login", use_container_width=True):
                    ok, res = authenticate_user("demo_user", "password123")
                    if ok:
                        st.session_state["user"] = res
                        st.rerun()

            st.caption("Tip: Click 'Instant Demo Login' to test with pre-saved movie ratings!")

        # --- REGISTER TAB ---
        with auth_tab_register:
            st.markdown("##### Create Your Personal Account")
            su_username = st.text_input("Choose Username", key="auth_su_user", placeholder="e.g. cinephile_99")
            su_password = st.text_input("Choose Password", type="password", key="auth_su_pwd", placeholder="At least 6 characters")
            su_email = st.text_input("Email (Optional)", key="auth_su_email", placeholder="you@example.com")

            all_genres = [
                "Action", "Adventure", "Animation", "Comedy", "Crime",
                "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", "Thriller"
            ]
            su_pref = st.multiselect(
                "Select Favorite Genres",
                all_genres,
                default=["Action", "Sci-Fi", "Adventure"],
                help="We use these to personalize your recommendations from day one."
            )

            if st.button("Create Account & Start 🎬", type="primary", use_container_width=True):
                if len(su_username.strip()) < 3:
                    st.error("Username must be at least 3 characters.")
                elif len(su_password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    pref_str = "|".join(su_pref)
                    ok, res = register_user(su_username, su_password, su_email, pref_str)
                    if ok:
                        st.session_state["user"] = res
                        st.success("Account created successfully!")
                        st.rerun()
                    else:
                        st.error(res)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        _, c_guest, _ = st.columns([1, 2, 1])
        with c_guest:
            if st.button("👀 Continue as Guest (Preview Catalog)", use_container_width=True):
                st.session_state["guest_mode"] = True
                st.rerun()

    # Stop execution here so no app contents render on the auth page
    st.stop()


# =========================================================================
# VIEW 2: MAIN CINEMA APP (Clean, User-Focused, Movie-First)
# =========================================================================

current_user = st.session_state.get("user")

# Top Navigation Bar
c_nav_left, c_nav_right = st.columns([3, 1])
with c_nav_left:
    st.markdown("### 🍿 CinemaAI")
with c_nav_right:
    if current_user:
        c_u1, c_u2 = st.columns([2, 1])
        with c_u1:
            st.markdown(f"**👤 {current_user['username']}**")
        with c_u2:
            if st.button("Sign Out", type="secondary", key="nav_sign_out"):
                st.session_state["user"] = None
                st.session_state["guest_mode"] = False
                st.rerun()
    else:
        if st.button("🔑 Sign In / Register", type="primary", key="nav_sign_in"):
            st.session_state["guest_mode"] = False
            st.rerun()

# Clean Navigation Tabs
tabs = st.tabs([
    "🎬 Recommended For You",
    "🔍 Browse & Search",
    "🔖 My Library & Ratings",
    "🧠 AI Architecture & Benchmarks"
])


# =========================================================================
# TAB 1: Recommended For You
# =========================================================================
with tabs[0]:
    if not current_user:
        st.info("💡 You are browsing as a Guest. Sign in to save movies and get hyper-personalized recommendations tailored to your taste!")
        user_ratings_count = 0
        u_id = 1  # Fallback to popular sample user
    else:
        u_id = current_user["id"]
        u_ratings_df = get_user_ratings(u_id)
        user_ratings_count = len(u_ratings_df)

    st.markdown("#### Top Movies Handpicked For You")

    c_filter, c_count = st.columns([3, 1])
    with c_filter:
        if current_user and current_user.get("preferred_genres"):
            st.caption(f"Curated based on your ratings and favorite genres: `{current_user['preferred_genres']}`")
        else:
            st.caption("Personalized using deep collaborative filtering and genre alignment.")
    with c_count:
        num_recs = st.select_slider("Show:", options=[6, 9, 12, 18], value=6, key="user_rec_count")

    with st.spinner("Finding the best movies for you..."):
        if current_user:
            recs_df, _ = hybrid.recommend_for_db_user(u_id, top_n=num_recs)
        else:
            recs_df, _ = hybrid.recommend_for_user(user_id=1, top_n=num_recs)

    if recs_df.empty:
        st.info("Rate a few titles in 'Browse & Search' to generate your personalized recommendations!")
    else:
        for i, row in recs_df.iterrows():
            m_id = int(row["movieId"])
            m_title = row["title"]
            genres_list = str(row["genres"]).split("|")
            in_watchlist = is_in_watchlist(u_id, m_id) if current_user else False

            poster_url = get_movie_poster(m_id, m_title, row["genres"])

            with st.container():
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                col_post, col_meta = st.columns([1, 4])

                with col_post:
                    st.image(poster_url, use_container_width=True)

                with col_meta:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <h3 style="margin: 0; padding: 0;">{m_title} <span style="font-size: 15px; color: #64748b;">({row['year']})</span></h3>
                            <div style="margin: 6px 0 10px 0;">
                                {' '.join([f'<span class="badge">{g}</span>' for g in genres_list])}
                            </div>
                        </div>
                        <span class="badge-match">⭐ {row['hybrid_score']:.1f} / 5.0 Match</span>
                    </div>
                    <p style="color: #cbd5e1; font-size: 14px; margin-bottom: 12px;">
                        <em>💡 {row['explanation']}</em>
                    </p>
                    """, unsafe_allow_html=True)

                    c_act1, c_act2 = st.columns([2, 1])

                    with c_act1:
                        if current_user:
                            star = st.select_slider(
                                f"Rate '{m_title}'",
                                options=[1.0, 2.0, 3.0, 4.0, 5.0],
                                value=4.0,
                                key=f"feed_rate_{m_id}"
                            )
                            if st.button("Save Rating ⭐", key=f"feed_save_{m_id}"):
                                save_user_rating(u_id, m_id, star, m_title)
                                st.success(f"Rated {star}⭐!")
                                st.rerun()
                        else:
                            st.caption("Sign in to rate.")

                    with c_act2:
                        if current_user:
                            if in_watchlist:
                                if st.button("Remove ❌", key=f"feed_rm_{m_id}"):
                                    remove_from_watchlist(u_id, m_id)
                                    st.rerun()
                            else:
                                if st.button("Bookmark 🔖", key=f"feed_add_{m_id}"):
                                    add_to_watchlist(u_id, m_id, m_title)
                                    st.success("Saved!")
                                    st.rerun()
                        else:
                            st.caption("Sign in to bookmark.")

                st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# TAB 2: Browse & Search
# =========================================================================
with tabs[1]:
    st.markdown("#### 🔍 Browse 9,700+ Movies")

    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        query = st.text_input("Search by title:", value="", placeholder="e.g. Inception, Dark Knight, Toy Story...")
    with col_s2:
        genre_filter = st.selectbox(
            "Filter by Genre:",
            ["All Genres"] + loader.unique_genres
        )

    all_movies = loader.movies_df.copy()
    if query:
        all_movies = all_movies[all_movies["title"].str.contains(query, case=False, na=False)]
    if genre_filter != "All Genres":
        all_movies = all_movies[all_movies["genres"].str.contains(genre_filter, case=False, na=False)]

    results_subset = all_movies.head(8)

    if results_subset.empty:
        st.warning("No movies found matching your search. Try another keyword.")
    else:
        st.caption(f"Showing {len(results_subset)} matching titles")
        for _, row in results_subset.iterrows():
            mid = int(row["movieId"])
            title = row["title"]
            genres_list = str(row["genres"]).split("|")
            poster_url = get_movie_poster(mid, title, row["genres"])
            in_w = is_in_watchlist(current_user["id"], mid) if current_user else False

            with st.container():
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                cp, cm = st.columns([1, 4])

                with cp:
                    st.image(poster_url, use_container_width=True)

                with cm:
                    st.markdown(f"""
                    <h3 style="margin: 0;">{title} <span style="font-size: 15px; color: #64748b;">({row['year']})</span></h3>
                    <div style="margin: 6px 0 10px 0;">
                        {' '.join([f'<span class="badge">{g}</span>' for g in genres_list])}
                    </div>
                    """, unsafe_allow_html=True)

                    cr1, cr2, cr3 = st.columns([2, 1, 1])
                    with cr1:
                        if current_user:
                            user_val = st.slider(f"Rate", 0.5, 5.0, 4.0, 0.5, key=f"browse_slider_{mid}")
                            if st.button("Submit Rating ⭐", key=f"browse_rate_{mid}"):
                                save_user_rating(current_user["id"], mid, user_val, title)
                                st.success(f"Saved {user_val}⭐!")
                                st.rerun()
                        else:
                            st.caption("Sign in to rate.")

                    with cr2:
                        if current_user:
                            if in_w:
                                if st.button("Remove ❌", key=f"browse_rm_{mid}"):
                                    remove_from_watchlist(current_user["id"], mid)
                                    st.rerun()
                            else:
                                if st.button("Bookmark 🔖", key=f"browse_add_{mid}"):
                                    add_to_watchlist(current_user["id"], mid, title)
                                    st.success("Bookmarked!")
                                    st.rerun()

                    with cr3:
                        with st.popover("Similar Titles"):
                            sim_df = content_model.get_similar_movies(mid, top_n=4)
                            for _, s_row in sim_df.iterrows():
                                st.markdown(f"- **{s_row['title']}** ({s_row['similarity_score']*100:.0f}% match)")

                st.markdown('</div>', unsafe_allow_html=True)


# =========================================================================
# TAB 3: My Library & Ratings
# =========================================================================
with tabs[2]:
    if not current_user:
        st.info("Sign in or create an account to view and manage your personal movie library.")
    else:
        u_id = current_user["id"]
        u_stats = get_user_profile_stats(u_id)

        st.markdown(f"#### 📁 Personal Library: {current_user['username']}")

        c_stat1, c_stat2, c_stat3 = st.columns(3)
        c_stat1.metric("⭐ Movies Rated", u_stats["rating_count"])
        c_stat2.metric("📊 Your Average Rating", f"{u_stats['avg_rating']} / 5.0")
        c_stat3.metric("🔖 Saved in Watchlist", u_stats["watchlist_count"])

        st.markdown("---")

        sub_tab1, sub_tab2 = st.tabs(["🔖 Watchlist", "⭐ Rated Movies"])

        with sub_tab1:
            w_items = get_user_watchlist(u_id)
            if not w_items:
                st.info("Your watchlist is empty. Bookmark movies in 'Browse' or 'Recommended' to watch them later!")
            else:
                w_ids = [w["movie_id"] for w in w_items]
                w_movies = loader.movies_df[loader.movies_df["movieId"].isin(w_ids)]

                for _, row in w_movies.iterrows():
                    mid = int(row["movieId"])
                    poster_url = get_movie_poster(mid, row["title"], row["genres"])

                    with st.container():
                        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                        c_wp, c_wm, c_wa = st.columns([1, 4, 1])

                        with c_wp:
                            st.image(poster_url, use_container_width=True)

                        with c_wm:
                            st.markdown(f"**{row['title']}** ({row['year']})")
                            st.caption(f"🎭 {row['genres']}")

                        with c_wa:
                            if st.button("Remove ❌", key=f"lib_rm_{mid}"):
                                remove_from_watchlist(u_id, mid)
                                st.rerun()

                        st.markdown('</div>', unsafe_allow_html=True)

        with sub_tab2:
            u_ratings_df = get_user_ratings(u_id)
            if u_ratings_df.empty:
                st.info("You haven't rated any movies yet.")
            else:
                merged = u_ratings_df.merge(loader.movies_df[["movieId", "title", "genres", "year"]], on="movieId")
                st.dataframe(
                    merged[["title", "genres", "rating", "year", "timestamp"]],
                    use_container_width=True
                )

                del_title = st.selectbox("Remove rating for:", merged["title"].values, key="lib_del_select")
                if st.button("Delete Rating", type="secondary"):
                    del_row = merged[merged["title"] == del_title].iloc[0]
                    delete_user_rating(u_id, int(del_row["movieId"]))
                    st.success(f"Removed rating for '{del_title}'.")
                    st.rerun()


# =========================================================================
# TAB 4: Technical AI & Benchmarks
# =========================================================================
with tabs[3]:
    st.markdown("#### 🧠 Recommendation Engine Architecture & Benchmarks")
    st.caption("Detailed technical specifications and empirical performance analysis.")

    # Architecture Overview
    st.markdown("""
```
                   ┌────────────────────────────────────────┐
                   │           Inputs (Batch, ...)          │
                   ├────────────────────┬───────────────────┤
                   │ User ID (1)        │ Movie ID (1)      │ Movie Genres (19-dim)
                   └────────┬───────────┴─────────┬─────────┴──────────┬────────────┘
                            │                     │                    │
            ┌───────────────┴───────────────┐     │                    ▼
            │                               │     │               Dense(32, ReLU)
            ▼                               ▼     ▼                    │
    GMF User Embedding             GMF Movie Embeddings                │
         (64-dim)                       (64-dim)                       │
            │                               │                          │
            └───────────────┬───────────────┘                          │
                            ▼                                          │
                   Element-wise Multiply                               │
                     (GMF Interaction)                                 │
                            │                                          │
            ┌───────────────┴───────────────┐                          │
            │                               │                          │
            ▼                               ▼                          ▼
    MLP User Embedding             MLP Movie Embeddings ──► Concatenate Layer
         (64-dim)                       (64-dim)                       │
                                                                       ▼
                                                          Dense(128, ReLU) + BN + Dropout
                                                                       │
                                                                       ▼
                                                          Dense(64, ReLU) + BN + Dropout
                                                                       │
                                                                       ▼
                                                          Dense(32, ReLU)
                                                                       │
                                                                       ▼
                                                         ┌───────────────────────────┐
                                                         │ NeuMF Concatenation Layer │
                                                         └─────────────┬─────────────┘
                                                                       ▼
                                                           Output Rating (0.5 - 5.0)
```
    """)

    st.markdown("---")

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        st.markdown("##### Prediction Accuracy (Lower is Better)")
        cf_m = summary.get("cf_metrics", {"rmse": 0.9294, "mae": 0.7168})
        dp_m = summary.get("deep_metrics", {"rmse": 0.8510, "mae": 0.6559})

        metric_df = pd.DataFrame({
            "Metric": ["RMSE", "MAE"],
            "Collaborative Filtering (SVD)": [cf_m["rmse"], cf_m["mae"]],
            "Deep Neural Network (NeuMF)": [dp_m["rmse"], dp_m["mae"]]
        }).set_index("Metric")

        st.bar_chart(metric_df)

    with c_b2:
        st.markdown("##### Sparsity Stress-Test (How Deep NN Handles Sparse Data)")
        if "sparsity_results" in summary:
            sp_df = pd.DataFrame(summary["sparsity_results"])
            st.dataframe(sp_df, use_container_width=True)

            chart_data = sp_df[["sparsity_bucket", "cf_rmse", "deep_rmse"]].set_index("sparsity_bucket")
            st.line_chart(chart_data)

    st.markdown("##### 2D Latent Embedding Space (PCA)")
    with st.spinner("Projecting latent movie embeddings..."):
        movie_embeddings = deep_net.extract_movie_embeddings()
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(movie_embeddings)

        popular_ids = loader.ratings_df["movieId"].value_counts().head(250).index
        sampled_rows = []

        for mid in popular_ids:
            midx = loader.movie_to_idx.get(mid)
            if midx is not None:
                minfo = loader.get_movie_by_id(mid)
                primary_genre = minfo["genres"].split("|")[0] if pd.notna(minfo["genres"]) else "Other"
                sampled_rows.append({
                    "title": minfo["title"],
                    "genre": primary_genre,
                    "pca_1": reduced[midx, 0],
                    "pca_2": reduced[midx, 1]
                })

        plot_df = pd.DataFrame(sampled_rows)

    st.scatter_chart(
        data=plot_df,
        x="pca_1",
        y="pca_2",
        color="genre",
        use_container_width=True
    )
