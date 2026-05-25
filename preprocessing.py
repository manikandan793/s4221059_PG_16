import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR   = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {"science": "#2196F3", "denial": "#F44336"}

MIN_WORDS            = 5
SPAM_LONG_THRESHOLD  = 3
SPAM_SHORT_THRESHOLD = 20
SPAM_LENGTH_CUTOFF   = 50


STOPWORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his", "she",
    "her", "it", "its", "they", "them", "their", "what", "which", "who",
    "this", "that", "these", "those", "am", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "just", "don", "should", "now", "would", "could",
    "really", "like", "think", "people", "thing", "way", "even", "get", "got",
    "one", "two", "youtube", "video", "comment", "edit", "deleted", "removed",
    "http", "https", "www", "com", "gt", "lt", "amp", "thats", "dont",
    "doesnt", "didnt", "wasnt", "isnt", "im", "ive", "youre", "theyre",
}

SCIENCE_KEYWORDS = {
    "ipcc", "emission", "emissions", "carbon", "temperature", "warming",
    "greenhouse", "fossil", "renewable", "consensus", "peer", "review",
    "study", "research", "scientist", "data", "evidence", "model", "arctic",
    "methane", "atmosphere", "climate", "ocean",
}

DENIAL_KEYWORDS = {
    "hoax", "scam", "fake", "agenda", "conspiracy", "corrupt", "globalist",
    "alarmist", "propaganda", "lie", "fraud", "manipulation", "hide", "cover",
    "secret", "natural", "cycle", "sun", "solar", "medieval", "climategate",
    "sheeple",
}

DELETED_USERS = {"[deleted]", "[removed]", "None", "nan", "", "[unknown]",
                 "AutoModerator"}
BOT_RE = re.compile(r"(bot|auto|automod)$", re.IGNORECASE)


def is_bad_author(name):
    if not isinstance(name, str) or name.strip() in DELETED_USERS:
        return True
    return bool(BOT_RE.search(name))


URL_RE      = re.compile(r"http\S+|www\.\S+")
MD_LINK_RE  = re.compile(r"\[(.*?)\]\(.*?\)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
HTML_ENT_RE = re.compile(r"&\w+;")
NON_ALPHA   = re.compile(r"[^a-zA-Z\s']")
WHITESPACE  = re.compile(r"\s+")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = HTML_ENT_RE.sub(" ", text)
    text = NON_ALPHA.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip().lower()


def lemmatise(word):
    if len(word) <= 4:
        return word
    for s in ("ingly", "edly", "ies", "ied", "ing", "ed", "es", "ly", "s"):
        if word.endswith(s) and len(word) - len(s) >= 3:
            return word[:-len(s)]
    return word


def tokenise(text):
    return [lemmatise(t) for t in text.split()
            if t not in STOPWORDS and len(t) > 2]


def smart_dedup(df, body_col):
    counts  = df[body_col].astype(str).value_counts()
    lengths = df.groupby(body_col)[body_col].first().astype(str).str.len()

    long_bodies  = set(lengths[lengths >  SPAM_LENGTH_CUTOFF].index)
    short_bodies = set(lengths[lengths <= SPAM_LENGTH_CUTOFF].index)

    spam = (set(counts[counts > SPAM_LONG_THRESHOLD ].index) & long_bodies) \
         | (set(counts[counts > SPAM_SHORT_THRESHOLD].index) & short_bodies)

    n0 = len(df)
    df = df[~df[body_col].isin(spam)].copy()
    print(f"    dedup: dropped {n0 - len(df):,} spam-style rows")
    return df


def compute_features(tokens, score):
    s = set(tokens)
    sci = len(s & SCIENCE_KEYWORDS)
    den = len(s & DENIAL_KEYWORDS)
    return {
        "word_count":           len(tokens),
        "unique_words":         len(s),
        "lexical_diversity":    len(s) / max(len(tokens), 1),
        "sci_keyword_count":    sci,
        "denial_keyword_count": den,
        "keyword_alignment":    ("science" if sci > den
                                 else "denial" if den > sci
                                 else "neutral"),
        "log_score":            float(np.log1p(max(score, 0))),
    }


def preprocess_comments(df):
    print("\n  Preprocessing comments")
    funnel = {"raw": len(df)}
    print(f"    raw rows: {len(df):,}")

    df = df[~df["author"].apply(is_bad_author)]
    funnel["after author filter"] = len(df)
    print(f"    after author filter: {len(df):,}")

    df = df[df["body"].fillna("").astype(str).str.split().apply(len) >= MIN_WORDS]
    funnel["after length filter"] = len(df)
    print(f"    after length filter: {len(df):,}")

    df = smart_dedup(df, "body")
    funnel["after dedup"] = len(df)

    df["text_clean"] = df["body"].apply(clean_text)
    df["tokens"]     = df["text_clean"].apply(tokenise)
    df["tokens_str"] = df["tokens"].apply(" ".join)
    df = df[df["tokens"].apply(len) >= 2].copy()
    funnel["after token filter"] = len(df)
    print(f"    after token filter:  {len(df):,}")

    feats = df.apply(lambda r: compute_features(
        r["tokens"], int(r.get("score", 0) or 0)), axis=1)
    df = pd.concat([df.reset_index(drop=True),
                    pd.DataFrame(list(feats))], axis=1)

    print(f"    final: {len(df):,} ({len(df) / max(funnel['raw'], 1):.1%} of raw)")
    return df, funnel


def preprocess_posts(df):
    print("\n  Preprocessing posts")
    n0 = len(df)
    print(f"    raw rows: {n0:,}")

    df["raw_text"] = (df["title"].fillna("") + " " +
                      df.get("selftext", pd.Series([""] * len(df))).fillna("")
                     ).str.strip()

    df = df[~df["author"].apply(is_bad_author)]
    df = df[df["raw_text"].str.split().apply(len) >= MIN_WORDS]
    df = smart_dedup(df, "raw_text")

    df["text_clean"] = df["raw_text"].apply(clean_text)
    df["tokens"]     = df["text_clean"].apply(tokenise)
    df["tokens_str"] = df["tokens"].apply(" ".join)
    df = df[df["tokens"].apply(len) >= 3].copy()

    feats = df.apply(lambda r: compute_features(
        r["tokens"], int(r.get("score", 0) or 0)), axis=1)
    df = pd.concat([df.reset_index(drop=True),
                    pd.DataFrame(list(feats))], axis=1)

    print(f"    final: {len(df):,} ({len(df)/max(n0,1):.1%} of raw)")
    return df


def plot_funnel(funnel):
    fig, ax = plt.subplots(figsize=(9, 5))
    stages, vals = list(funnel.keys()), list(funnel.values())
    bars = ax.barh(stages, vals, color="#2196F3")
    for bar, v in zip(bars, vals):
        ax.text(v, bar.get_y() + bar.get_height()/2,
                f"  {v:,}", va="center", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Number of comments")
    ax.set_title("Preprocessing funnel — comments")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/preprocessing_funnel.png", dpi=150,
                bbox_inches="tight")
    plt.close()


def plot_dashboard(comments_df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for camp, c in PALETTE.items():
        d = comments_df[comments_df["camp"] == camp]["word_count"]
        axes[0, 0].hist(d, bins=40, alpha=0.6, label=camp, color=c)
    axes[0, 0].set_title("Word count per comment, by camp")
    axes[0, 0].set_xlabel("Words")
    axes[0, 0].set_ylabel("Comments")
    axes[0, 0].set_xlim(0, comments_df["word_count"].quantile(0.99))
    axes[0, 0].legend()

    counts = comments_df["camp"].value_counts()
    axes[0, 1].bar(counts.index, counts.values,
                   color=[PALETTE.get(c, "gray") for c in counts.index])
    for i, v in enumerate(counts.values):
        axes[0, 1].text(i, v, f"{v:,}", ha="center", va="bottom")
    axes[0, 1].set_title("Comments by camp")
    axes[0, 1].set_ylabel("Count")

    sci = comments_df["sci_keyword_count"]
    den = comments_df["denial_keyword_count"]
    axes[1, 0].scatter(sci + np.random.normal(0, 0.05, len(sci)),
                       den + np.random.normal(0, 0.05, len(sci)),
                       c=comments_df["camp"].map(PALETTE),
                       alpha=0.25, s=8)
    axes[1, 0].set_xlabel("Science-vocab keywords")
    axes[1, 0].set_ylabel("Denial-vocab keywords")
    axes[1, 0].set_title("Keyword vocabulary per comment")
    axes[1, 0].set_xlim(-0.5, 8); axes[1, 0].set_ylim(-0.5, 8)

    for camp, c in PALETTE.items():
        d = comments_df[comments_df["camp"] == camp]["lexical_diversity"]
        if len(d) > 10:
            sns.kdeplot(d, ax=axes[1, 1], label=camp, color=c, fill=True,
                        alpha=0.3, linewidth=2)
    axes[1, 1].set_title("Lexical diversity per comment")
    axes[1, 1].set_xlabel("Unique words / total words")
    axes[1, 1].legend()

    plt.suptitle("Preprocessing summary — cleaned corpus", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/preprocessing_dashboard.png", dpi=150,
                bbox_inches="tight")
    plt.close()


def main():
    
    print("STEP 2 — PREPROCESSING")

    comments_path = f"{DATA_DIR}/raw_comments.csv"
    posts_path    = f"{DATA_DIR}/raw_posts.csv"

    if not os.path.exists(comments_path):
        sys.exit(f"  Missing {comments_path}. Run 01b_youtube_to_csv.py first.")

    comments = pd.read_csv(comments_path)
    clean_comments, funnel = preprocess_comments(comments)
    clean_comments.to_csv(f"{DATA_DIR}/clean_comments.csv", index=False)

    clean_posts = None
    if os.path.exists(posts_path):
        posts = pd.read_csv(posts_path)
        clean_posts = preprocess_posts(posts)
        clean_posts.to_csv(f"{DATA_DIR}/clean_posts.csv", index=False)

    plot_funnel(funnel)
    plot_dashboard(clean_comments)


    print(f"  Clean comments : {len(clean_comments):,}")
    if clean_posts is not None:
        print(f"  Clean posts    : {len(clean_posts):,}")
    print(f"  Unique users   : {clean_comments['author'].nunique():,}")
    print(f"  Unique videos  : {clean_comments['post_id'].nunique():,}")
    print("\n  Comments by camp:")
    print(clean_comments.groupby("camp").size().to_string())



if __name__ == "__main__":
    main()
