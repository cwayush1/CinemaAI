# 🍿 Hybrid Movie Recommender System

> **A production-grade hybrid recommendation system leveraging Collaborative Filtering, Content-Based Filtering, and Deep Neural Networks (TensorFlow/Keras) to deliver accurate, personalized movie recommendations even under extreme data sparsity.**

---

## 📌 Project Overview & Highlights

This project implements an end-to-end recommendation engine trained on the **MovieLens** dataset (100,836 ratings across 9,742 movies and 610 users). It addresses the fundamental challenges in recommender systems: **user-item matrix sparsity (98.3%)** and the **cold-start problem**.

### Key Resume Highlights Demonstrated
- **Hybrid Recommendation Architecture:** Blends **Collaborative Filtering** (Matrix Factorization via TruncatedSVD), **Content-Based Filtering** (TF-IDF + Cosine Similarity on genres and user tags), and a **Deep Neural Network** (Neural Collaborative Filtering / NeuMF).
- **Data Engineering with Pandas & NumPy:** Processed high-dimensional rating matrices, engineered contiguous 0-indexed embedding lookups, multi-hot genre representations, and sparse CSR matrices.
- **Deep Neural Networks in TensorFlow & Keras:** Designed a two-tower Neural CF model featuring a Generalized Matrix Factorization (GMF) branch, a Multi-Layer Perceptron (MLP) branch with content embedding injection, Batch Normalization, Dropout, and L2 regularization to counteract extreme matrix sparsity.
- **Empirical Validation:** Demonstrated an **8.4% improvement in RMSE (0.8510 vs. 0.9294)** over classical matrix factorization, with up to **9.35% error reduction** across user interaction tiers.
- **Interactive Web Application:** Built a comprehensive Streamlit dashboard featuring real-time personalized recommendations, latent space PCA visualization, and an interactive cold-start sandbox.

---

## 🏗️ System Architecture

```
                             ┌───────────────────────────────────────┐
                             │       User & Item Interactions        │
                             │ (100k+ Ratings, 9.7k Movies, 610 Users)│
                             └──────────────────┬────────────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
    ┌─────────────────────────┐   ┌───────────────────────────┐   ┌──────────────────────────┐
    │  Content-Based Engine   │   │  Collaborative Engine     │   │   Deep Neural Network    │
    │  (Scikit-Learn TF-IDF)  │   │  (TruncatedSVD Latent)    │   │    (TensorFlow NeuMF)    │
    ├─────────────────────────┤   ├───────────────────────────┤   ├──────────────────────────┤
    │ • Genre & Tag features  │   │ • User mean centering     │   │ • GMF & MLP dual towers  │
    │ • Sublinear TF-IDF      │   │ • 40 Latent dimensions    │   │ • Content dense injection│
    │ • User profile vectors  │   │ • User & Item factors     │   │ • Dropout & BatchNorm    │
    │ • Cosine similarity     │   │ • Global mean fallback    │   │ • Embeddings regularizer │
    └────────────┬────────────┘   └─────────────┬─────────────┘   └────────────┬─────────────┘
                 │                              │                              │
                 │ Score: S_content             │ Score: S_cf                  │ Score: S_deep
                 └──────────────────────┐       │       ┌──────────────────────┘
                                        ▼       ▼       ▼
                             ┌───────────────────────────────────────┐
                             │     Dynamic Hybrid Fusion Engine      │
                             │     S = w_deep*S_deep + w_cf*S_cf     │
                             │           + w_content*S_content       │
                             │   (Weights adapt based on sparsity)   │
                             └──────────────────┬────────────────────┘
                                                │
                                                ▼
                             ┌───────────────────────────────────────┐
                             │     Top-N Ranked Recommendations      │
                             │     + Human-Readable Explanations     │
                             └───────────────────────────────────────┘
```

---

## 🧮 Mathematical Formulations

### 1. Content-Based Filtering (TF-IDF & User Profile)
For each movie $i$, its genres and tags are tokenized into a normalized TF-IDF vector $\mathbf{v}_i \in \mathbb{R}^V$. For user $u$ with rated movies $M_u$ and ratings $r_{u,i}$:
$$\mathbf{p}_u = \frac{\sum_{i \in M_u} (r_{u,i} - \bar{r}_u) \mathbf{v}_i}{\sum_{i \in M_u} |r_{u,i} - \bar{r}_u|}, \quad S_{\text{content}}(u, i) = \frac{\mathbf{p}_u \cdot \mathbf{v}_i}{\|\mathbf{p}_u\|_2 \|\mathbf{v}_i\|_2}$$

### 2. Collaborative Filtering (TruncatedSVD Matrix Factorization)
The centered rating matrix $R_{u,i} - \bar{r}_u \approx U \Sigma V^T$:
$$\hat{r}_{\text{cf}}(u, i) = \bar{r}_u + \mathbf{p}_u^{\text{SVD}} \cdot \mathbf{q}_i^{\text{SVD}}$$

### 3. Deep Neural Collaborative Filtering (NeuMF with Content Injection)
The network fuses a linear interaction branch (GMF) and a non-linear deep interaction branch (MLP):
$$\mathbf{\phi}^{\text{GMF}} = \mathbf{e}_u^{\text{GMF}} \odot \mathbf{e}_i^{\text{GMF}}$$
$$\mathbf{h}_0^{\text{MLP}} = [\mathbf{e}_u^{\text{MLP}}, \mathbf{e}_i^{\text{MLP}}, \text{ReLU}(W_g \mathbf{g}_i + b_g)]$$
$$\mathbf{h}_l^{\text{MLP}} = \text{Dropout}(\text{BatchNorm}(\text{ReLU}(W_l \mathbf{h}_{l-1}^{\text{MLP}} + b_l)))$$
$$\hat{y}(u, i) = \mathbf{w}^T [\mathbf{\phi}^{\text{GMF}}, \mathbf{h}_L^{\text{MLP}}, b_u, b_i] + \mu$$

### 4. Dynamic Sparsity-Adaptive Weighting
When evaluating user $u$ with rating history size $N_u$:
- **Cold-Start ($N_u = 0$):** $w_{\text{content}} = 1.0, w_{\text{deep}} = 0.0, w_{\text{cf}} = 0.0$
- **Sparse User ($1 \le N_u < 15$):** $w_{\text{content}} = 0.55, w_{\text{deep}} = 0.30, w_{\text{cf}} = 0.15$
- **Dense/Warm User ($N_u \ge 15$):** $w_{\text{deep}} = 0.50, w_{\text{cf}} = 0.30, w_{\text{content}} = 0.20$

---

## 📊 Benchmark & Evaluation Results

Evaluated on a held-out 20% test split (20,168 ratings):

### Overall Rating Prediction Accuracy
| Model Architecture | RMSE (Lower is better) | MAE (Lower is better) | Relative RMSE Gain |
| :--- | :---: | :---: | :---: |
| **Collaborative Filtering (TruncatedSVD)** | 0.9294 | 0.7168 | Baseline |
| **Deep Neural Network (NeuMF + Content)** | **0.8510** | **0.6559** | **+8.44% Improvement** |

### Sparsity Stress-Test (Performance Across User Density Tiers)
| User Sparsity Tier | Test Samples | SVD RMSE | Deep NN RMSE | Deep Error Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Sparse (< 30 ratings)** | 965 | 0.9944 | **0.9414** | **+5.33%** |
| **Moderate (30 - 100 ratings)** | 3,308 | 0.9443 | **0.8944** | **+5.29%** |
| **Dense (> 100 ratings)** | 15,895 | 0.9221 | **0.8359** | **+9.35%** |

*Notice: Traditional matrix factorization error degrades to near 1.0 on sparse users, whereas the regularized Deep Neural Network with content embeddings maintains superior accuracy.*

---

## 📂 Project Structure

```
movie_recommender_system/
├── data/
│   ├── download_data.py         # Automated MovieLens downloader & fallback generator
│   ├── movies.csv               # 9,742 movie titles, release years, and genres
│   ├── ratings.csv              # 100,836 user ratings (0.5 to 5.0 stars)
│   └── tags.csv                 # 3,683 user-assigned keyword tags
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Parsing, 0-indexing, genre multi-hot, CSR sparse matrix
│   ├── content_based.py         # Scikit-Learn TF-IDF, cosine similarity, user profiles
│   ├── collaborative.py         # TruncatedSVD matrix factorization & item similarity
│   ├── neural_cf.py             # TensorFlow / Keras NeuMF with genre content embeddings
│   ├── hybrid.py                # Adaptive sparsity fusion & explanation generator
│   └── evaluate.py              # RMSE, MAE, Precision@K, Recall@K, NDCG@K, sparsity tiers
│
├── saved_models/
│   ├── deep_recommender.weights.h5 # Trained neural network weights
│   ├── artifacts.pkl             # Serialized SVD factors, mappings & TF-IDF vectors
│   └── evaluation_summary.json  # Precomputed evaluation metrics
│
├── tests/
│   ├── __init__.py
│   └── test_recommender.py      # Unit & integration test suite (6/6 passing)
│
├── train.py                     # Full end-to-end training and evaluation pipeline
├── evaluate_all.py              # Comprehensive benchmark reporting tool
├── run_demo.py                  # Command-line recommendation and similarity demo
├── app.py                       # Interactive Streamlit dashboard
└── requirements.txt             # Project dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation
Ensure Python 3.10+ is installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Download Data & Train Models
Run the end-to-end training pipeline. It will automatically download the MovieLens dataset, train all three models, evaluate benchmarks, and save the artifacts:
```bash
python train.py
```

### 3. Run the Unit Test Suite
Verify that all algorithms and components function properly:
```bash
python -m unittest tests/test_recommender.py
```

### 4. Interactive Command-Line Demo
Get top 5 recommendations for user ID 1:
```bash
python run_demo.py --user 1 --top_n 5
```
Or find movies similar in content to *Star Wars (movieId 260)*:
```bash
python run_demo.py --movie 260 --top_n 5
```

### 5. Launch the Web Application
Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```

---

## 🖥️ Streamlit App Features

1. **Personalized Recommendations:** Select any user to see their favorite titles, top hybrid recommendations, and clear explanations of why each title was recommended.
2. **Content Discovery:** Search any movie to find similar titles using TF-IDF and tag analysis.
3. **Deep Learning Latent Explorer:** View the NeuMF architecture diagram and an interactive 2D PCA scatter plot of learned latent movie embeddings.
4. **Benchmarks & Sparsity Analysis:** Compare RMSE and MAE across algorithms and user sparsity buckets.
5. **Interactive Cold-Start Sandbox:** Rate 5 popular movies in the UI and generate instant custom recommendations.
