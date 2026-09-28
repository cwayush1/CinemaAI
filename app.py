"""
CinemaAI Movie Recommender — Full Application (app.py)
Fully Integrated Dark UI (Fixes CSS path issues and giant Streamlit image bugs)
"""

import os
import sys
import json
import joblib
import pandas as pd
import streamlit as st

# =========================================================================
# 1. PAGE CONFIG & INLINE CSS (Guarantees UI loads perfectly)
# =========================================================================
st.set_page_config(
    page_title="CinemaAI | Movie Recommender",
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Force Dark Background and Font */
.stApp {
    background-color: #0B0F19 !important;
    background-image: radial-gradient(at 50% 0%, rgba(30, 41, 59, 1) 0px, transparent 100%) !important;
    color: #F1F5F9 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Hide Default Streamlit Chrome */
#MainMenu, footer, header { visibility: hidden; height: 0; }
.block-container { max-width: 1200px !important; padding: 2rem !important; }

/* FIX THE GIANT MOVIE POSTERS */
div[data-testid="stImage"] {
    display: flex;
    justify-content: center;
    align-items: flex-start;
}
div[data-testid="stImage"] img {
    max-width: 160px !important;
    min-width: 150px !important;
    height: 240px !important;
    object-fit: cover !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.movie-card {
    background: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.movie-card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
}

.stat-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
}

.navbar-glass {
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 16px 24px;
    margin-bottom: 30px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Badges */
.badge-match {
    background: #10B981; color: white;
    padding: 4px 12px; border-radius: 20px;
    font-size: 13px; font-weight: 700;
}
.badge-genre {
    background: rgba(99, 102, 241, 0.15); color: #818CF8;
    padding: 4px 10px; border-radius: 6px;
    font-size: 12px; font-weight: 600; margin-right: 6px;
    border: 1px solid rgba(99, 102, 241, 0.3);
    display: inline-block;
    margin-bottom: 6px;
}

/* Fix Streamlit Sliders & Buttons */
.stSlider { padding-bottom: 15px !important; }
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: 0.2s !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
    color: white !important;
    border: none !important;
}
.stTextInput > div > div > input, .stSelectbox > div > div {
    background-color: rgba(15, 23, 42, 0.75) !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important;
}
/* CRAZY GLOWING STAR RATINGS */
div[data-testid="stFeedback"] {
    display: flex;
    gap: 4px;
    margin-top: 5px;
    padding-bottom: 15px;
}
div[data-testid="stFeedback"] button {
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
div[data-testid="stFeedback"] button:hover {
    transform: scale(1.4) translateY(-3px) !important;
    filter: drop-shadow(0 0 12px rgba(251, 191, 36, 0.9)) !important;
}
div[data-testid="stFeedback"] svg {
    width: 24px !important;
    height: 24px !important;
    color: #FBBF24 !important;
    fill: transparent; /* Empty stars */
}
/* Fill in stars when clicked */
div[data-testid="stFeedback"] button[data-checked="true"] svg {
    fill: #FBBF24 !important;
    filter: drop-shadow(0 0 6px rgba(251, 191, 36, 0.5));
}
/* CRAZY GLOWING SLIDER STYLING */
.stSlider {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}
/* The main track */
.stSlider div[data-testid="stTickBar"] {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 10px !important;
    height: 6px !important;
}
/* The active track part */
.stSlider div[data-testid="stSliderTickBar"] > div {
    background: linear-gradient(135deg, #6366F1 0%, #C084FC 100%) !important;
}
/* The thumb (the circle you drag) */
.stSlider div[role="slider"] {
    background: linear-gradient(135deg, #6366F1 0%, #C084FC 100%) !important;
    box-shadow: 0 0 15px rgba(192, 132, 252, 0.6) !important;
    border: 2px solid #FFFFFF !important;
    width: 20px !important;
    height: 20px !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
/* Hover effect on thumb */
.stSlider div[role="slider"]:hover, .stSlider div[role="slider"]:focus {
    transform: scale(1.3) !important;
    box-shadow: 0 0 25px rgba(192, 132, 252, 0.9) !important;
    outline: none !important;
}
/* The text value bubble above the slider */
.stSlider div[data-testid="stThumbValue"] {
    color: #F8FAFC !important;
    background: rgba(15, 23, 42, 0.9) !important;
    border: 1px solid rgba(192, 132, 252, 0.4) !important;
    padding: 4px 10px !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
}
/* Min/Max text on the sides */
.stSlider div[data-testid="stTickBarMin"], .stSlider div[data-testid="stTickBarMax"] {
    color: #475569 !important;
    font-weight: 600 !important;
}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)


# =========================================================================
# 2. BACKEND & MACHINE LEARNING INITIALIZATION
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.database import (
    init_db, register_user, authenticate_user, save_user_rating,
    delete_user_rating, get_user_ratings, add_to_watchlist,
    remove_from_watchlist, get_user_watchlist, is_in_watchlist,
    get_user_profile_stats, create_user_session, get_user_by_session_token,
    delete_user_session
)
from src.data_loader import MovieDataLoader
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.neural_cf import DeepRecommenderNet, HAS_TF
from src.hybrid import HybridRecommender
from src.poster_fetcher import get_movie_poster

init_db()

@st.cache_resource(show_spinner=False)
def load_all_models_and_data():
    loader = MovieDataLoader()
    loader.load_data()
    loader.split_train_test()

    saved_dir = os.path.join(BASE_DIR, "saved_models")
    artifacts_path = os.path.join(saved_dir, "artifacts.pkl")
    weights_path = os.path.join(saved_dir, "deep_recommender.weights.h5")

    if not os.path.exists(artifacts_path) or not os.path.exists(weights_path):
        from train import run_pipeline
        run_pipeline(epochs=5, batch_size=256)

    artifacts = joblib.load(artifacts_path)

    content_model = ContentBasedRecommender(loader)
    content_model.vectorizer = artifacts["tfidf_vectorizer"]
    content_model.tfidf_matrix = artifacts["tfidf_matrix"]
    content_model.movie_idx_to_pos = artifacts["movie_idx_to_pos"]
    content_model.pos_to_movie_idx = artifacts["pos_to_movie_idx"]

    cf_model = CollaborativeRecommender(loader, n_components=40)
    cf_model.user_means = artifacts["cf_user_means"]
    cf_model.user_factors = artifacts["cf_user_factors"]
    cf_model.item_factors = artifacts["cf_item_factors"]
    cf_model.reconstructed_matrix = artifacts["cf_reconstructed"]

    num_users = len(loader.user_to_idx)
    num_movies = len(loader.movie_to_idx)
    genre_dim = loader.movie_genre_matrix.shape[1]

    deep_net = None
    if HAS_TF and os.path.exists(weights_path):
        try:
            deep_net = DeepRecommenderNet(num_users, num_movies, genre_dim, embedding_dim=32)
            deep_net.load(weights_path)
        except Exception:
            deep_net = None

    hybrid = HybridRecommender(loader, content_model, cf_model, deep_net)
    
    return loader, content_model, cf_model, deep_net, hybrid

loader, content_model, cf_model, deep_net, hybrid = load_all_models_and_data()

# Session State & Persistence
if "user" not in st.session_state:
    st.session_state["user"] = None
if "guest_mode" not in st.session_state:
    st.session_state["guest_mode"] = False

# Auto-restore session from URL query parameters on page refresh (F5)
if st.session_state["user"] is None:
    session_token = st.query_params.get("session")
    if session_token:
        restored_user = get_user_by_session_token(session_token)
        if restored_user:
            st.session_state["user"] = restored_user
        else:
            # Token invalid or removed from database, clear query params
            st.query_params.clear()
    elif st.query_params.get("guest") == "1":
        st.session_state["guest_mode"] = True

# =========================================================================
# 3. VIEW 1: AUTHENTICATION PORTAL
# =========================================================================
if st.session_state["user"] is None and not st.session_state["guest_mode"]:
    _, col_auth, _ = st.columns([1, 2, 1])

    with col_auth:
        st.markdown("""
        <div style="text-align: center; margin-top: 40px; margin-bottom: 40px;">
            <div style="display: inline-block; background: rgba(99,102,241,0.15); padding: 16px 24px; border-radius: 20px; border: 1px solid rgba(99,102,241,0.3); margin-bottom: 24px; box-shadow: 0 8px 32px rgba(99,102,241,0.2);">
                <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="url(#cinemaai-grad)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <defs>
                        <linearGradient id="cinemaai-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="#818CF8" />
                            <stop offset="100%" stop-color="#C084FC" />
                        </linearGradient>
                    </defs>
                    <rect x="2" y="2" width="20" height="20" rx="4"></rect>
                    <line x1="18" y1="2" x2="18" y2="22"></line>
                    <line x1="6" y1="2" x2="6" y2="22"></line>
                    <line x1="2" y1="6" x2="6" y2="6"></line>
                    <line x1="2" y1="10" x2="6" y2="10"></line>
                    <line x1="2" y1="14" x2="6" y2="14"></line>
                    <line x1="2" y1="18" x2="6" y2="18"></line>
                    <line x1="18" y1="6" x2="22" y2="6"></line>
                    <line x1="18" y1="10" x2="22" y2="10"></line>
                    <line x1="18" y1="14" x2="22" y2="14"></line>
                    <line x1="18" y1="18" x2="22" y2="18"></line>
                    <polygon points="10 9 15 12 10 15 10 9" fill="rgba(192,132,252,0.3)"></polygon>
                </svg>
            </div>
            <h1 style="font-size: 48px; margin: 0;"><span class="brand-gradient">CinemaAI</span></h1>
            <p style="color: #94A3B8; font-size: 16px; font-weight: 500; margin-top: 12px; letter-spacing: 0.5px;">Hybrid Neural Recommendation System</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        auth_tab_sign_in, auth_tab_register = st.tabs(["🔑 Sign In", "📝 Create Account"])

        with auth_tab_sign_in:
            st.markdown("<br>", unsafe_allow_html=True)
            si_username = st.text_input("Username", key="auth_si_user")
            si_password = st.text_input("Password", type="password", key="auth_si_pwd")
            st.markdown("<br>", unsafe_allow_html=True)
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("Sign In 🚀", type="primary", use_container_width=True):
                    ok, res = authenticate_user(si_username, si_password)
                    if ok:
                        token = create_user_session(res["id"])
                        st.query_params["session"] = token
                        st.session_state["user"] = res
                        st.rerun()
                    else:
                        st.error(res)
            with c_btn2:
                if st.button("⚡ Demo Account", use_container_width=True):
                    ok, res = authenticate_user("demo_user", "password123")
                    if ok:
                        token = create_user_session(res["id"])
                        st.query_params["session"] = token
                        st.session_state["user"] = res
                        st.rerun()

        with auth_tab_register:
            st.markdown("<br>", unsafe_allow_html=True)
            su_username = st.text_input("Choose Username", key="auth_su_user")
            su_password = st.text_input("Choose Password", type="password", key="auth_su_pwd")
            su_email = st.text_input("Email Address", key="auth_su_email")
            su_pref = st.multiselect("Favorite Genres", loader.unique_genres, default=["Action", "Sci-Fi"])
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account 🎬", type="primary", use_container_width=True):
                ok, res = register_user(su_username, su_password, su_email, "|".join(su_pref))
                if ok:
                    token = create_user_session(res["id"])
                    st.query_params["session"] = token
                    st.session_state["user"] = res
                    st.rerun()
                else:
                    st.error(res)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("👀 Continue as Guest (Explore Catalog)", use_container_width=True):
            st.session_state["guest_mode"] = True
            st.query_params["guest"] = "1"
            st.rerun()
    st.stop()
# =========================================================================
# 4. VIEW 2: MAIN CINEMA APP
# =========================================================================
current_user = st.session_state.get("user")

# Navbar
st.markdown('<div class="navbar-glass">', unsafe_allow_html=True)
c_nav_left, c_nav_right = st.columns([3, 1])
with c_nav_left:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 18px;">
        <div style="background: rgba(99,102,241,0.15); padding: 12px 16px; border-radius: 16px; border: 1px solid rgba(99,102,241,0.3); box-shadow: 0 4px 12px rgba(99,102,241,0.2);">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="url(#nav-grad)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <defs><linearGradient id="nav-grad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#818CF8" /><stop offset="100%" stop-color="#C084FC" /></linearGradient></defs>
                <rect x="2" y="2" width="20" height="20" rx="4"></rect>
                <line x1="18" y1="2" x2="18" y2="22"></line>
                <line x1="6" y1="2" x2="6" y2="22"></line>
                <line x1="2" y1="6" x2="6" y2="6"></line><line x1="2" y1="10" x2="6" y2="10"></line><line x1="2" y1="14" x2="6" y2="14"></line><line x1="2" y1="18" x2="6" y2="18"></line>
                <line x1="18" y1="6" x2="22" y2="6"></line><line x1="18" y1="10" x2="22" y2="10"></line><line x1="18" y1="14" x2="22" y2="14"></line><line x1="18" y1="18" x2="22" y2="18"></line>
                <polygon points="10 9 15 12 10 15 10 9" fill="rgba(192,132,252,0.3)"></polygon>
            </svg>
        </div>
        <div>
            <h3 style="margin: 0;"><span class="nav-brand-text">CinemaAI</span></h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
with c_nav_right:
    if current_user:
        if st.button("Sign Out", type="secondary"):
            token = st.query_params.get("session")
            if token:
                delete_user_session(token)
            st.query_params.clear()
            st.session_state["user"] = None
            st.session_state["guest_mode"] = False
            st.rerun()
    else:
        if st.button("🔑 Sign In", type="primary"):
            st.query_params.clear()
            st.session_state["guest_mode"] = False
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

tabs = st.tabs(["🎬 Recommended For You", "🔍 Browse & Search Catalog", "🔖 My Library & Watchlist"])

# -------------------------------------------------------------------------
# TAB 1: Recommended For You
# -------------------------------------------------------------------------
with tabs[0]:
    if not current_user:
        st.info("💡 You are browsing as a Guest. Sign in to personalize recommendations and save movies to your watchlist!")
    
    u_id = current_user["id"] if current_user else 1
    
    c_head1, c_head2 = st.columns([3, 1])
    with c_head1:
        st.markdown("### Top Movie Recommendations")
    with c_head2:
        st.markdown("<div style='margin-bottom: 5px; color: #94A3B8; font-size: 13px; font-weight: 600;'>Display Count</div>", unsafe_allow_html=True)
        # Replaces the slider with sleek clickable toggle buttons
        num_recs = st.segmented_control("Display Count", options=[6, 9, 12, 18], default=6, label_visibility="collapsed")
        
        # Fallback just in case they unclick the button
        if num_recs is None:
            num_recs = 6
    with st.spinner("Analyzing neural embeddings..."):
        recs_df, _ = hybrid.recommend_for_db_user(u_id, top_n=num_recs) if current_user else hybrid.recommend_for_user(1, num_recs)

    if recs_df.empty:
        st.warning("Rate a few titles in 'Browse & Search' to generate your personalized AI recommendations!")
    else:
        for i, row in recs_df.iterrows():
            mid = int(row["movieId"])
            genres_list = str(row["genres"]).split("|")
            poster_url = get_movie_poster(mid, row["title"], row["genres"])
            in_watchlist = is_in_watchlist(u_id, mid) if current_user else False
            
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            col_post, col_meta = st.columns([1, 4])
            with col_post:
                st.image(poster_url)
            with col_meta:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <h3 style="margin: 0; color: white;">{row['title']} <span style="font-size: 15px; color: #94A3B8;">({row['year']})</span></h3>
                        <div style="margin: 8px 0 12px 0;">{' '.join([f'<span class="badge-genre">{g}</span>' for g in genres_list])}</div>
                    </div>
                    <span class="badge-match">⭐ {row['hybrid_score']:.1f} / 5.0 Match</span>
                </div>
                <p style="color: #CBD5E1; font-size: 14px; background: rgba(15,23,42,0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                    💡 <strong>Why AI matched this:</strong> {row['explanation']}
                </p>
                """, unsafe_allow_html=True)
                
                if current_user:
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown("<span style='font-size: 13px; color: #94A3B8;'>Rate this movie:</span>", unsafe_allow_html=True)
                        
                        # Native 1-click stars (returns 0-4)
                        rating_index = st.feedback("stars", key=f"feed_r_{mid}")
                        
                        if rating_index is not None:
                            star_val = float(rating_index + 1)
                            # Session state check prevents infinite loops/double saves
                            if st.session_state.get(f"saved_rec_{mid}") != star_val:
                                save_user_rating(u_id, mid, star_val, row['title'])
                                st.session_state[f"saved_rec_{mid}"] = star_val
                                st.toast(f"Saved {star_val}⭐ for {row['title']}!", icon="✅")
                    with c2:
                        if in_watchlist:
                            if st.button("Remove ❌", key=f"feed_rm_{mid}"):
                                remove_from_watchlist(u_id, mid)
                                st.rerun()
                        else:
                            if st.button("Bookmark 🔖", key=f"feed_b_{mid}"):
                                add_to_watchlist(u_id, mid, row['title'])
                                st.success("Saved!")
                                st.rerun()
                else:
                    st.caption("Sign in to submit ratings and bookmarks.")
            st.markdown('</div>', unsafe_allow_html=True)


# -------------------------------------------------------------------------
# TAB 2: Browse & Search Catalog
# -------------------------------------------------------------------------
with tabs[1]:
    st.markdown("### 🔍 Search & Explore Catalog")

    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        query = st.text_input("Search movie title:", placeholder="e.g. Inception, Dark Knight...")
    with col_s2:
        genre_filter = st.selectbox("Filter Genre:", ["All Genres"] + loader.unique_genres)

    all_movies = loader.movies_df.copy()
    if query:
        all_movies = all_movies[all_movies["title"].str.contains(query, case=False, na=False)]
    if genre_filter != "All Genres":
        all_movies = all_movies[all_movies["genres"].str.contains(genre_filter, case=False, na=False)]

    results_subset = all_movies.head(8)

    if results_subset.empty:
        st.warning("No movies matched your search parameters.")
    else:
        st.caption(f"Showing top {len(results_subset)} matching titles")
        for _, row in results_subset.iterrows():
            mid = int(row["movieId"])
            title = row["title"]
            genres_list = str(row["genres"]).split("|")
            poster_url = get_movie_poster(mid, title, row["genres"])
            in_w = is_in_watchlist(current_user["id"], mid) if current_user else False

            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            cp, cm = st.columns([1, 4])
            with cp:
                st.image(poster_url)
            with cm:
                st.markdown(f"""
                <h3 style="margin: 0; color: white;">{title} <span style="font-size: 15px; color: #94A3B8;">({row['year']})</span></h3>
                <div style="margin: 10px 0 16px 0;">{' '.join([f'<span class="badge-genre">{g}</span>' for g in genres_list])}</div>
                """, unsafe_allow_html=True)

                cr1, cr2, cr3 = st.columns([2, 1, 1])
                with cr1:
                    if current_user:
                        st.markdown("<span style='font-size: 13px; color: #94A3B8;'>Rate this title:</span>", unsafe_allow_html=True)
                        
                        rating_index = st.feedback("stars", key=f"br_{mid}")
                        
                        if rating_index is not None:
                            star_val = float(rating_index + 1)
                            if st.session_state.get(f"saved_cat_{mid}") != star_val:
                                save_user_rating(current_user["id"], mid, star_val, title)
                                st.session_state[f"saved_cat_{mid}"] = star_val
                                st.toast(f"Saved {star_val}⭐ for {title}!", icon="✅")
                with cr2:
                    if current_user:
                        if in_w:
                            if st.button("Remove ❌", key=f"br_rm_{mid}"):
                                remove_from_watchlist(current_user["id"], mid)
                                st.rerun()
                        else:
                            if st.button("Save  🔖", key=f"br_add_{mid}"):
                                add_to_watchlist(current_user["id"], mid, title)
                                st.success("Bookmarked!")
                                st.rerun()
                with cr3:
                    with st.popover("Similar Titles"):
                        sim_df = content_model.get_similar_movies(mid, top_n=4)
                        for _, s_row in sim_df.iterrows():
                            st.markdown(f"- **{s_row['title']}** ({s_row['similarity_score']*100:.0f}% match)")
            st.markdown('</div>', unsafe_allow_html=True)


# -------------------------------------------------------------------------
# TAB 3: My Library & Watchlist
# -------------------------------------------------------------------------
with tabs[2]:
    if not current_user:
        st.info("Please sign in or register to access and manage your personal ratings and watchlist.")
    else:
        u_id = current_user["id"]
        u_stats = get_user_profile_stats(u_id)

        st.markdown("### 📁 Personal Movie Library")

        c_stat1, c_stat2, c_stat3 = st.columns(3)
        with c_stat1:
            st.markdown(f'<div class="stat-card"><h2 style="margin:0; font-size: 2.5rem; color: #F8FAFC;">{u_stats["rating_count"]}</h2><p style="margin:0; color: #94A3B8;">⭐ Movies Rated</p></div>', unsafe_allow_html=True)
        with c_stat2:
            st.markdown(f'<div class="stat-card"><h2 style="margin:0; font-size: 2.5rem; color: #F8FAFC;">{u_stats["avg_rating"]}</h2><p style="margin:0; color: #94A3B8;">📊 Average Score</p></div>', unsafe_allow_html=True)
        with c_stat3:
            st.markdown(f'<div class="stat-card"><h2 style="margin:0; font-size: 2.5rem; color: #F8FAFC;">{u_stats["watchlist_count"]}</h2><p style="margin:0; color: #94A3B8;">🔖 Saved Watchlist</p></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        sub_tab1, sub_tab2 = st.tabs(["🔖 Saved Watchlist", "⭐ Rated Titles History"])

        with sub_tab1:
            w_items = get_user_watchlist(u_id)
            if not w_items:
                st.info("Your watchlist is currently empty. Bookmark titles in the catalog to review them here.")
            else:
                w_ids = [w["movie_id"] for w in w_items]
                w_movies = loader.movies_df[loader.movies_df["movieId"].isin(w_ids)]

                for _, row in w_movies.iterrows():
                    mid = int(row["movieId"])
                    poster_url = get_movie_poster(mid, row["title"], row["genres"])
                    
                    st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                    c_wp, c_wm, c_wa = st.columns([1, 4, 1])
                    with c_wp:
                        st.image(poster_url)
                    with c_wm:
                        st.markdown(f"<h3 style='margin:0; color: white;'>{row['title']} <span style='font-size:15px; color:#94A3B8;'>({row['year']})</span></h3>", unsafe_allow_html=True)
                        st.caption(f"Genres: {row['genres']}")
                    with c_wa:
                        if st.button("Remove ❌", key=f"lib_rm_{mid}"):
                            remove_from_watchlist(u_id, mid)
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

        with sub_tab2:
            u_ratings_df = get_user_ratings(u_id)
            if u_ratings_df.empty:
                st.info("You have not rated any movies yet.")
            else:
                merged = u_ratings_df.merge(loader.movies_df[["movieId", "title", "genres", "year"]], on="movieId")
                st.dataframe(merged[["title", "genres", "rating", "year", "timestamp"]], use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                del_title = st.selectbox("Remove rating for specific title:", merged["title"].values, key="lib_del_select")
                if st.button("Delete Rating Entry", type="secondary"):
                    del_row = merged[merged["title"] == del_title].iloc[0]
                    delete_user_rating(u_id, int(del_row["movieId"]))
                    st.success(f"Removed rating entry for '{del_title}'.")
                    st.rerun()
# =========================================================================
# 5. PROFESSIONAL FOOTER
# =========================================================================
st.markdown("""
<div style="margin-top: 80px; padding: 40px 20px 24px 20px; background: linear-gradient(180deg, transparent 0%, rgba(15, 23, 42, 0.8) 100%); border-top: 1px solid rgba(255, 255, 255, 0.05); display: flex; flex-direction: column; align-items: center; text-align: center; border-radius: 16px;">
<div style="display: flex; align-items: center; justify-content: center; gap: 10px; margin-bottom: 20px;">
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="url(#foot-grad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <defs><linearGradient id="foot-grad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#818CF8" /><stop offset="100%" stop-color="#C084FC" /></linearGradient></defs>
    <rect x="2" y="2" width="20" height="20" rx="4"></rect>
    <line x1="18" y1="2" x2="18" y2="22"></line><line x1="6" y1="2" x2="6" y2="22"></line>
    <line x1="2" y1="6" x2="6" y2="6"></line><line x1="2" y1="10" x2="6" y2="10"></line><line x1="2" y1="14" x2="6" y2="14"></line><line x1="2" y1="18" x2="6" y2="18"></line>
    <line x1="18" y1="6" x2="22" y2="6"></line><line x1="18" y1="10" x2="22" y2="10"></line><line x1="18" y1="14" x2="22" y2="14"></line><line x1="18" y1="18" x2="22" y2="18"></line>
    <polygon points="10 9 15 12 10 15 10 9" fill="rgba(192,132,252,0.3)"></polygon>
</svg>
<h3 style="margin: 0; font-size: 22px; font-weight: 800; background: linear-gradient(135deg, #818CF8 0%, #C084FC 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.5px;">CinemaAI</h3>
</div>
<div style="color: #94A3B8; font-size: 14px; font-weight: 500; margin-bottom: 24px; display: flex; gap: 24px; flex-wrap: wrap; justify-content: center;">
<span style="cursor: pointer;">About Us</span> 
<span style="cursor: pointer;">API Documentation</span> 
<span style="cursor: pointer;">Privacy Policy</span> 
<span style="cursor: pointer;">Terms of Service</span>
</div>
<div style="color: #475569; font-size: 13px; line-height: 1.6;">
© 2026 CinemaAI Engine. Built with Neural Collaborative Filtering & Content-Based Search.<br>
Crafted for next-generation cinematic discovery.
</div>
</div>
""", unsafe_allow_html=True)