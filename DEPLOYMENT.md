# 🚀 CinemaAI Deployment Guide

This guide provides step-by-step instructions to deploy the **CinemaAI Hybrid Movie Recommender System** to the cloud for free with a permanent, public HTTPS URL.

---

## 🌟 Method 1: Streamlit Community Cloud (Recommended — 100% Free Forever)

Streamlit Community Cloud is the official, easiest, and fastest hosting platform for Streamlit applications. It automatically pulls from your GitHub repository, provides continuous integration (redeploys whenever you push changes), and gives you a custom URL (e.g., `https://cinemaai-recommender.streamlit.app`).

### Step 1: Push Your Code to GitHub

1. Open your browser and log into [GitHub](https://github.com).
2. Click **New Repository** (or visit [github.com/new](https://github.com/new)):
   - **Repository name**: `movie-recommender-system` (or any name you prefer)
   - **Visibility**: Public (required for free Streamlit Community Cloud hosting)
   - Do **NOT** initialize with a README, .gitignore, or license (these are already configured in this repo).
   - Click **Create repository**.

3. In PowerShell / Terminal on your machine, run these commands:
   ```powershell
   cd C:\Users\ayush\OneDrive\Desktop\movie_recommender_system
   git add .
   git commit -m "Configure production deployment and session persistence"
   git branch -M main
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/movie-recommender-system.git
   git push -u origin main
   ```
   *(Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username)*

### Step 2: Deploy on Streamlit Cloud

1. Visit [share.streamlit.io](https://share.streamlit.io/) and click **Continue with GitHub**.
2. Click **Create app** (or **New app**).
3. Fill in the deployment details:
   - **Repository**: `YOUR_GITHUB_USERNAME/movie-recommender-system`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL** (Optional): Choose a custom subdomain (e.g. `cinemaai-movies.streamlit.app`)
4. Click **Deploy!** 🚀

Streamlit Cloud will install dependencies from `requirements.txt`, load the trained neural and SVD models from `saved_models/`, and your application will be live across the world in 2–3 minutes!

---

## 🤗 Method 2: Hugging Face Spaces (Free 16 GB RAM Tier)

Hugging Face Spaces is another excellent option tailored for AI/ML applications, offering 16 GB RAM for free.

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Fill in:
   - **Space name**: `cinemaai-recommender`
   - **License**: `mit`
   - **Space SDK**: **Streamlit**
   - **Space hardware**: `CPU basic • 2 vCPU • 16 GB • Free`
3. Clone the space repo or connect your GitHub repository.
4. Push your files — Hugging Face will automatically build and launch the app at `https://huggingface.co/spaces/YOUR_USERNAME/cinemaai-recommender`.

---

## ☁️ Method 3: Render (Web Service)

1. Sign up at [Render.com](https://render.com/).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository.
4. Render will automatically detect `render.yaml` or you can manually configure:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Select the **Free** instance type and click **Create Web Service**.

---

## 🐳 Method 4: Docker Container

To run or deploy using Docker anywhere (AWS, GCP, DigitalOcean, local server):

1. **Build the container image**:
   ```bash
   docker build -t cinemaai-recommender .
   ```

2. **Run the container**:
   ```bash
   docker run -d -p 8501:8501 --name cinemaai cinemaai-recommender
   ```

3. Open `http://localhost:8501` in your browser.

---

## 🔒 Configuration & Persistence Notes

- **Persistent Sessions**: SQLite handles session tokens and personal ratings in `data/recommender.db`.
- **Pretrained Weights**: The saved models in `saved_models/` (`artifacts.pkl` and `deep_recommender.weights.h5`) are committed and loaded directly in production.
- **Port & Theme**: Production defaults are declared in `.streamlit/config.toml`.
