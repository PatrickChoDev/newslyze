# Newslyze

An end-to-end NLP pipeline that scrapes AI-related news from The Guardian, discovers subtopics via unsupervised clustering, scores sentiment, and forecasts coverage trends over the next 3 months.

## Pipeline

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `01_scrape_guardian.ipynb` | Fetch AI-tagged articles from The Guardian API → `data/raw_articles.csv` |
| 2 | `02_classify_subtopics.ipynb` | BERTopic clustering (SBERT → UMAP → HDBSCAN) to discover 10–20 subtopics → `data/classified_articles.csv` |
| 3 | `03_sentiment_scoring.ipynb` | Score each article with VADER or RoBERTa → `data/scored_articles.csv` |
| 4 | `04_engineer_timeseries.ipynb` | Aggregate into weekly time series per subtopic → `data/timeseries.csv` |
| 5 | `05_mining_analysis.ipynb` | Trend analysis, anomaly detection, cross-correlation |
| 6 | `06_cluster_others.ipynb` | Topic quality review and model refinement (diagnostic) |
| 7 | `07_forecast_subtopics.ipynb` | Prophet + SARIMA forecasts per subtopic → `data/forecasts.csv` |

Subtopics discovered include: AI & Jobs, AI Regulation, AI in Healthcare, AI Safety, Generative AI, AI & Big Tech.

## Setup

Requires Python 3.13+. Dependencies are managed with [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

You also need a Guardian API key. Set it as an environment variable before running notebook 01:

```bash
export GUARDIAN_API_KEY=your_key_here
```

Get a free key at [open-platform.theguardian.com](https://open-platform.theguardian.com/access/).

## Running the Pipeline

Run notebooks in order (01 → 07). Each notebook reads from and writes to the `data/` directory.

```bash
uv run jupyter notebook
```

## Key Dependencies

- **BERTopic** — topic modelling with transformer embeddings
- **sentence-transformers** — `all-mpnet-base-v2` for article embeddings
- **UMAP + HDBSCAN** — dimensionality reduction and density clustering
- **VADER / transformers** — sentiment scoring
- **Prophet + pmdarima** — time series forecasting
