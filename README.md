# Personalized Recommendation System

**Author:** Rekha Sakipalli
**Project Objective:** Build a reusable, modular recommender system implementing collaborative, content-based, and hybrid approaches for personalized recommendations. Provide code, evaluation, and a simple web demo (Flask/Streamlit).

---

## Summary

This project demonstrates a full-cycle recommender pipeline:

* Week 1 — Data acquisition & cleaning (MovieLens, Amazon reviews examples)
* Week 2 — Collaborative filtering (user-based, item-based, matrix factorization)
* Week 3 — Content-based and hybrid models (TF–IDF on metadata + hybridize with collaborative signals)
* Week 4 — Model comparison, dashboard, and demo web app

It includes training, evaluation (RMSE, MAE, Precision@K), and a small demo where users can get Top-N recommendations and simulate feedback (clicks) to re-rank recommendations.

---

## Datasets

Suggested example datasets (store under `data/`):

* **MovieLens** (e.g., `ml-latest-small/`): ratings, movies metadata.
* **Amazon Reviews** (subset): product reviews, product metadata.

> Place original dataset files in `data/` and update config paths in `config.yaml` or `notebooks/config.py`.

---

## Project Structure (recommended)

```
personalized-recsys/
├─ data/                      # raw and processed datasets
├─ notebooks/                 # EDA + experiments
├─ src/
│  ├─ data_preprocessing.py
│  ├─ collaborative.py        # user/item CF + matrix factorization wrappers
│  ├─ content_based.py        # TF-IDF and metadata pipelines
│  ├─ hybrid.py               # blending/cascading/hybrid logic
│  ├─ metrics.py              # RMSE/MAE/Precision@K implementations
│  ├─ train.py                # orchestration script to train models
│  └─ serve.py                # Flask/Streamlit demo app
├─ models/                    # saved model artifacts
├─ requirements.txt
├─ README.md
└─ LICENSE
```

---

## Week-by-week Implementation Plan

### Week 1 — Data Preparation

* Download MovieLens and a subset of Amazon reviews.
* EDA: rating distributions, missing metadata, cold-start analysis.
* Cleaning:

  * Fill or drop missing values as appropriate.
  * Normalize ratings (e.g., min-max or z-score) if using algorithms that require it.
  * Create train/validation/test splits (time-aware split if timestamps are available).
* Output: `data/processed/*.csv` and sampling scripts in `notebooks/`.

### Week 2 — Collaborative Filtering Models

* **User-based CF:** compute cosine similarity between user rating vectors; predict via weighted average.
* **Item-based CF:** item co-occurrence / item similarity and neighborhood-based predictions.
* **Matrix Factorization:**

  * SVD (e.g., using `surprise` SVD or implicit ALS)
  * NMF (non-negative matrix factorization) for interpretability
* **Libraries:** `scikit-surprise`, `lightfm`, or `implicit` for scaled MF.
* **Evaluation:** RMSE, MAE for rating prediction; Precision@K for top-N.

### Week 3 — Content-Based + Hybrid

* Extract item features: genre, title, description, product categories, tags.
* Use TF–IDF vectorization (scikit-learn) on item textual metadata.
* Content-based recommendations via cosine similarity between user profile vector and item vectors (user profile constructed from items the user interacted with).
* **Hybrid ideas:**

  * Weighted blending: α*collaborative_score + (1−α)*content_score
  * Cascading: use CF to shortlist, re-rank with content and business rules
  * Feedback loop (simple RL-inspired re-ranking): re-rank recommendations based on recent clicks/implicit feedback (e.g., apply higher weight to clicked item features)
* Implement an API to recompute re-ranking quickly after user feedback.

### Week 4 — Dashboard + Final Report

* Train all models on same splits; compare via RMSE/MAE/Precision@K/Recall@K/NDCG.
* Build a small demo app:

  * **Flask** for a simple backend + HTML UI, or **Streamlit** for fast interactive demo.
  * API endpoints: `/recommend/<user_id>?k=10`, `/feedback` (accepts clicks to update session-level ranking).
* Create a report summarizing methods, results, and recommendations for productionization.

---

## Evaluation Metrics

* **Prediction error:** RMSE, MAE
* **Ranking / Top-N:** Precision@K, Recall@K, NDCG
* Track also: coverage, novelty, and diversity for practical insights.

---

## Tools & Libraries

* Python 3.9+
* Core: `numpy`, `pandas`, `scikit-learn`, `scipy`
* Recommender-specific: `scikit-surprise`, `lightfm`, `implicit` (choose as needed)
* Web demo: `Flask` or `streamlit`
* Optional: `tensorflow`/`pytorch` if experimenting with deep models
* Visualization: `matplotlib`, `seaborn` (for notebooks/reports)
* Packaging: `pip`, `venv`, or `conda`

Example `requirements.txt` (minimal):

```
numpy
pandas
scikit-learn
scikit-surprise
lightfm
flask
streamlit
matplotlib
```

---

## How to run (quickstart)

1. Create environment and install:

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

2. Prepare data:

* Place datasets in `data/` and run preprocessing:

```bash
python src/data_preprocessing.py --input data/raw --output data/processed
```

3. Train a baseline CF model:

```bash
python src/train.py --model cf_user --config config.yaml
```

4. Evaluate:

```bash
python src/train.py --evaluate --model cf_user --config config.yaml
```

5. Run demo (Flask example):

```bash
python src/serve.py   # then open http://localhost:5000
```

Or start Streamlit:

```bash
streamlit run src/serve_streamlit.py
```

---

## Example commands for GitHub

* Initialize repo and push:

```bash
git init
git add .
git commit -m "Initial recommender system scaffold"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

---

## Tips & Notes

* Start small: use `ml-latest-small` for fast iteration.
* Log experiments (hyperparams, metrics) — use `mlflow` or simple CSV logs.
* For Precision@K, make sure to evaluate on held-out interactions (top-N offline protocols).
* Consider user & item cold-start strategies (fallback to content-based).
* Keep preprocessing deterministic and document random seeds.

---

## Deliverables

* `notebooks/` containing EDA and experiments
