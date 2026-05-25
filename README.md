# Climate Misinformation vs Scientific Consensus on YouTube

**Course:** COSC 2671 — Social Media and Network Analysis
**Assignment:** Assignment 2
**Team size:** 3
**Group Number:** PG 16

## 1. Project overview

This project investigates how climate-change misinformation communities
and scientific-consensus communities differ on YouTube — in their network
structure, influential users, community organisation, and discourse.
We construct **directed weighted reply networks** per camp from YouTube
comment data and compare them across structural, centrality, community,
sentiment, topical, and classification dimensions.

### Research question

> *How do climate-misinformation and scientific-consensus communities on
> YouTube differ structurally and discursively — in network topology,
> influential users, community organisation, sentiment, and topical
> framing — and what do those differences reveal about how each kind of
> content spreads?*


## 2. Data

### Source

**YouTube Data API v3** (`https://developers.google.com/youtube/v3`).
Public videos and comments only. No private data or PII.

### Collection

- Collected via `fetchYoutubeData_climate.py`.
- Date of collection: May 2026.
- Two query sets (10 each):
  - **Science-leaning**: e.g. "IPCC climate report explained", "global warming peer reviewed research".
  - **Denial-leaning**: e.g. "climate change hoax exposed", "global warming not real".
- Up to 50 top videos per query (by view count, English-language).
- Up to 70 top comments per video plus up to 10 first-level replies per thread.

### Collected dataset

| Item | Count |
|---|---|
| Videos | 687 (391 science / 296 denial) |
| Comments | 83,486 |
| Unique commenters | ~70,000 |

### Fields used

`youtubeDataDump.json` contains:

- **Video**: `videoId`, `camp`, `query`, `title`, `channelTitle`, `publishedAt`, `description`, `viewCount`, `likeCount`, `commentCount`.
- **Comment**: `commentId`, `author`, `replyTo`, `replyToType`, `text`, `likeCount`, `publishedAt`, `depth`.

`replyTo` is the key field that enables a **directed** reply network: A → B when A's comment replies to B's comment (or B's video, for top-level comments).

### How the data supports the analysis

- **Network analysis** — `replyTo` directly encodes user → user reply edges; `camp` labels each node's source community.
- **NLP analysis** — `text` provides the raw discourse for sentiment scoring, topic modelling, framing analysis, and classification.

### Limitations

- YouTube's `order=relevance` returns only the most-engaging comments, not a random sample.
- The dataset is a snapshot in time; comments are added and removed continuously.
- Some comments are missing replies because reply chains are truncated by the API.
- Camp labelling is based on the search query that returned the video; channels can post mixed-stance content.

## 3. Pipeline

```
project/
├── youtubeClient_climate.py        # API client setup
├── fetchYoutubeData_climate.py     # data collection → youtubeDataDump.json
├── 01b_youtube_to_csv.py           # JSON → raw_posts.csv, raw_comments.csv
├── 02_preprocessing.py             # cleaning, dedup, tokenisation, features
├── 03_network_analysis.py          # graphs, centrality, communities, E-I, propagation
├── 04_nlp_analysis.py              # RoBERTa sentiment, LDA, framing, classifier
├── 05_integrated_results.py        # cross-analysis + executive summary
├── run_all_youtube.py              # one-command runner for steps 1b → 5
│
├── requirements.txt
├── README.md
│
├── data/                           # all generated CSVs
└── outputs/                        # all generated PNG figures
```

## 4. How to run

### Setup

```bash
pip install -r requirements.txt
```

Set your YouTube API key (get one from <https://console.cloud.google.com>):

```bash
# Windows
set YOUTUBE_API_KEY= 'API_Key'

# macOS / Linux
export YOUTUBE_API_KEY= 'API_Key'
```

Optional: set a Hugging Face token for faster RoBERTa downloads
(get one from <https://huggingface.co/settings/tokens>):

```bash
# Windows
set HF_TOKEN= 'hf_xxxxxxxxxxxx'

# macOS / Linux
export HF_TOKEN= 'hf_xxxxxxxxxxxx'
```

### Run

Option 1 — Collect fresh YouTube data and run the full pipeline

```bash
python fetchYoutubeData_climate.py    # 10–20 min, collects YouTube data
python run_all_youtube.py             # runs steps 1b → 5 sequentially
```

Option 2 — Use the included dataset directly (skip data collection)

The repository already includes the collected dataset:
``` data/youtubeDataDump.json ```

This file contains all previously collected videos, comments, replies,
and metadata used throughout the analysis pipeline.

To reproduce the full analysis without recollecting data from the
YouTube API, simply skip the data-collection step and run:

``` bash 
python run_all_youtube.py 
```

Or run each stage individually:

```bash
python youtube_to_csv.py
python preprocessing.py
python network_analysis.py
python nlp_analysis.py
python integrated_results.py
```

### Approximate runtime (one full pass)

| Step | Time (GPU) | Time (CPU only) |
|---|---|---|
| Data fetch | 10–20 min (API rate-limited) | same |
| Preprocessing | 2–5 min | 2–5 min |
| Network analysis | 1–5 min | 1–5 min |
| NLP analysis | 5–10 min | 25–45 min |
| Integration | 30 sec | 30 sec |

## 5. Network design

| Element | Choice | Justification |
|---|---|---|
| Nodes | YouTube users (commenters + channels) | Users are the social actors |
| Edges | A → B if A replied to B's comment or video | Reply has a clear initiator and recipient |
| Direction | Directed | Replying is asymmetric: A speaks to B |
| Weight | Number of reply interactions | Captures intensity of repeated interaction |
| Filtering | min edge weight = 1, drop deleted/bot accounts | Removes spam and ghost activity |

## 6. Methods

### Network analysis

- Network-level metrics: density, clustering, components, diameter, reciprocity.
- Centrality: in/out-degree, PageRank, betweenness (k=300 sample), closeness, eigenvector.
- Composite influence score (0.4·PageRank + 0.3·betweenness + 0.2·degree + 0.1·closeness).
- Community detection: Louvain (modularity-optimising).
- **Krackhardt E-I index** (beyond-class) — quantifies echo-chamber tendency.
- **Influence propagation cascade** (beyond-class) — simulates spread from top hubs.
- Cross-camp bridge users with lean classification.

### NLP analysis 

- **Sentiment**: RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`), GPU-accelerated when available; per-comment probabilities for negative/neutral/positive plus a derived compound score.
- **Topic modelling**: LDA via gensim with c_v coherence-based K selection over K ∈ {5, 8, 10}.
- **Framing**: science-vs-denial keyword vocabulary intensity per camp.
- **Classifier**: TF-IDF (1–2 grams) + Logistic Regression on balanced classes; reports accuracy, macro F1, ROC-AUC, confusion matrix, and top discriminative words.

### Integration

- Per-user sentiment merged with centrality scores.
- Community-level sentiment profiles.
- Bridge-user sentiment by lean.
- Executive summary figure.

## 7. Outputs

### Data files (in `data/`)

`raw_posts.csv`, `raw_comments.csv`, `clean_posts.csv`, `clean_comments.csv`,
`network_stats.csv`, `centrality.csv`, `communities.csv`, `bridge_users.csv`,
`influence_propagation.csv`, `comments_sentiment.csv`, `topics.csv`,
`framing.csv`, `classifier_metrics.json`, `user_sentiment.csv`,
`centrality_sentiment.csv`, `community_sentiment.csv`, `bridge_sentiment.csv`,
`results_summary.csv`.


## 8. Assignment compliance

| Requirement | Where covered |
|---|---|
| What nodes represent | README §5 + step 3 docstring |
| What edges represent | README §5 + step 3 docstring |
| Directed / undirected, why | Directed — reply is asymmetric |
| Weighted, why | Weighted — captures repeated interaction |
| Filtering decisions | README §5 + step 2 + step 3 |
| Network measure (≥ 1) | Six: PageRank, betweenness, closeness, eigenvector, Louvain, E-I |
| PG: roles and information flow | Step 3 (communities + bridges + propagation) + step 5 |
| NLP component | Step 4 (RoBERTa + LDA + framing + classifier) |
| Beyond-class techniques | E-I index, influence propagation cascade, coherence-tuned LDA |


## 9. Sample data

A representative 10 MB sample of `youtubeDataDump.json` is included in the
submission to demonstrate the data structure used by the pipeline. The
full collection can be reproduced by setting a YouTube API key and
running `fetchYoutubeData_climate.py`.
