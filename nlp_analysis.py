import os
import sys
import json
import time
import ast
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, accuracy_score,
                              f1_score, confusion_matrix, roc_auc_score,
                              roc_curve)

warnings.filterwarnings("ignore")


DATA_DIR   = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {"science": "#2196F3", "denial": "#F44336", "bridge": "#4CAF50",
           "neutral": "#9E9E9E", "positive": "#4CAF50", "negative": "#E53935"}

plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "figure.dpi":        100,
})

ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
HF_TOKEN      = os.getenv("Your_Token_here", None)
BATCH_SIZE    = 64
MAX_LENGTH    = 256
SAMPLE_SIZE   = None

LDA_K_RANGE = (5, 8, 10)


SCIENCE_KEYWORDS = {
    "ipcc", "emission", "emissions", "carbon", "temperature", "warming",
    "greenhouse", "fossil", "renewable", "consensus", "peer", "review",
    "study", "research", "scientist", "data", "evidence", "model", "arctic",
    "methane", "atmosphere",
}

DENIAL_KEYWORDS = {
    "hoax", "scam", "fake", "agenda", "conspiracy", "corrupt", "globalist",
    "alarmist", "propaganda", "lie", "fraud", "manipulation", "hide", "cover",
    "secret", "natural", "cycle", "sun", "solar", "medieval", "climategate",
}


def tokens_from_row(row):
    v = row.get("tokens")
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith("["):
        return ast.literal_eval(v)
    return v.split() if isinstance(v, str) else []


# SENTIMENT — RoBERTa

class RobertaSentiment:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"      loading {ROBERTA_MODEL} on {self.device}")
        if HF_TOKEN:
            print("      using HF_TOKEN for authenticated requests")

        self.tokenizer = AutoTokenizer.from_pretrained(
            ROBERTA_MODEL, token=HF_TOKEN)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            ROBERTA_MODEL, token=HF_TOKEN, use_safetensors=True
        ).to(self.device).eval()
        self.labels = ["negative", "neutral", "positive"]

    def score_batch(self, texts):
        enc = self.tokenizer(texts, padding=True, truncation=True,
                              max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def score(self, texts):
        n = len(texts)
        out = np.zeros((n, 3), dtype=np.float32)
        t0 = time.time()
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch = [t if isinstance(t, str) and t.strip() else " "
                     for t in texts[start:end]]
            out[start:end] = self.score_batch(batch)
            if (start // BATCH_SIZE) % 20 == 0:
                elapsed = time.time() - t0
                rate = (end / elapsed) if elapsed > 0 else 0
                eta  = (n - end) / max(rate, 1)
                print(f"      {end:,}/{n:,}  "
                      f"({rate:.0f}/s, ETA {eta/60:.1f} min)", flush=True)
        return out


def sentiment_analysis(df):
    print("\n  SENTIMENT (RoBERTa)")
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"      sub-sampling {SAMPLE_SIZE:,} comments")
        df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"      scoring {len(df):,} comments")

    rb = RobertaSentiment()
    probs = rb.score(df["body"].fillna("").astype(str).tolist())

    df = df.copy()
    df["sent_neg"]      = probs[:, 0]
    df["sent_neu"]      = probs[:, 1]
    df["sent_pos"]      = probs[:, 2]
    df["sent_compound"] = probs[:, 2] - probs[:, 0]
    df["sent_label"]    = pd.Categorical(
        [rb.labels[i] for i in probs.argmax(axis=1)],
        categories=["negative", "neutral", "positive"]
    )
    df["sent_model"] = "roberta"

    print("\n      mean compound score by camp:")
    print(df.groupby("camp")["sent_compound"]
            .agg(["mean", "median", "std", "count"]).round(3).to_string())
    print("\n      label distribution by camp:")
    print(pd.crosstab(df["camp"], df["sent_label"], normalize="index")
            .round(3).to_string())
    return df


def plot_sentiment_dashboard(df):
    fig = plt.figure(figsize=(15, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

    camps = df["camp"].unique().tolist()
    cam_colors = [PALETTE[c] for c in camps]

    ax = fig.add_subplot(gs[0, 0])
    for camp in camps:
        d = df[df["camp"] == camp]["sent_compound"]
        if len(d) > 10:
            sns.kdeplot(d, ax=ax, label=camp.title(), color=PALETTE[camp],
                        fill=True, alpha=0.35, linewidth=2)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Sentiment compound score")
    ax.set_ylabel("Density")
    ax.set_title("Sentiment distribution by camp")
    ax.legend(frameon=True)

    ax = fig.add_subplot(gs[0, 1])
    sns.violinplot(data=df, x="camp", y="sent_compound", ax=ax,
                   palette={c: PALETTE[c] for c in camps}, inner="quartile",
                   linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(""); ax.set_ylabel("Compound score")
    ax.set_title("Sentiment shape by camp (violin)")

    ax = fig.add_subplot(gs[0, 2])
    ct = pd.crosstab(df["camp"], df["sent_label"], normalize="index")
    ct = ct.reindex(columns=["negative", "neutral", "positive"]).fillna(0)
    bottoms = np.zeros(len(ct))
    label_colors = {"negative": PALETTE["negative"],
                    "neutral":  PALETTE["neutral"],
                    "positive": PALETTE["positive"]}
    for col in ct.columns:
        ax.bar(ct.index, ct[col], bottom=bottoms,
               label=col, color=label_colors[col], edgecolor="white")
        for i, val in enumerate(ct[col]):
            if val > 0.05:
                ax.text(i, bottoms[i] + val/2, f"{val:.0%}",
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
        bottoms += ct[col].values
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.set_title("Sentiment-label proportions")
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    stats = df.groupby("camp")["sent_compound"].agg(
        ["mean", "std", "count"]).reindex(camps)
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats["ci"] = 1.96 * stats["se"]
    bars = ax.bar(stats.index, stats["mean"], yerr=stats["ci"],
                  color=cam_colors, capsize=6, edgecolor="white",
                  error_kw={"ecolor": "black", "linewidth": 1.5})
    for bar, m, n in zip(bars, stats["mean"], stats["count"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                m + (0.005 if m >= 0 else -0.015),
                f"{m:+.3f}\n(n={int(n):,})", ha="center",
                va="bottom" if m >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean compound (±95% CI)")
    ax.set_title("Mean sentiment by camp")

    ax = fig.add_subplot(gs[1, 1])
    neg_pct = (df.assign(is_neg=(df["sent_label"] == "negative").astype(int))
                 .groupby("camp")["is_neg"].mean().reindex(camps))
    bars = ax.bar(neg_pct.index, neg_pct.values, color=cam_colors,
                  edgecolor="white")
    for bar, v in zip(bars, neg_pct.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.1%}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, neg_pct.max() * 1.25)
    ax.set_ylabel("Fraction of comments")
    ax.set_title("Negative-sentiment fraction by camp")

    ax = fig.add_subplot(gs[1, 2])
    for camp in camps:
        d = df[df["camp"] == camp].sample(
            min(3000, (df["camp"] == camp).sum()), random_state=42
        )
        ax.scatter(d["sent_neg"], d["sent_pos"], alpha=0.25, s=8,
                   color=PALETTE[camp], label=camp.title())
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("P(negative)")
    ax.set_ylabel("P(positive)")
    ax.set_title("Per-comment probability landscape")
    ax.legend(frameon=True, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle("Sentiment analysis — RoBERTa", fontsize=15, y=0.995)
    plt.savefig(f"{OUTPUT_DIR}/sentiment_dashboard.png",
                dpi=160, bbox_inches="tight")
    plt.close()


# LDA TOPIC MODELLING

def lda_per_camp(df):
    print("\n  TOPIC MODELLING (LDA per camp)")
    results = {}

    for camp in ["science", "denial"]:
        sub = df[df["camp"] == camp]
        if len(sub) < 200:
            print(f"      {camp}: only {len(sub)} comments — skipped")
            continue

        token_lists = sub.apply(tokens_from_row, axis=1).tolist()
        token_lists = [t for t in token_lists if len(t) >= 3]
        if len(token_lists) < 200:
            continue

        dictionary = corpora.Dictionary(token_lists)
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        corpus = [dictionary.doc2bow(t) for t in token_lists]
        print(f"      {camp}: vocab={len(dictionary)}, docs={len(corpus):,}")

        best_k, best_score, best_model = None, -1, None
        scores = {}
        for k in LDA_K_RANGE:
            t0 = time.time()
            model = models.LdaModel(corpus=corpus, id2word=dictionary,
                                    num_topics=k, random_state=42,
                                    passes=5, iterations=50,
                                    alpha="auto", eta="auto")
            sc = CoherenceModel(model=model, texts=token_lists,
                                 dictionary=dictionary,
                                 coherence="c_v").get_coherence()
            scores[k] = sc
            print(f"        K={k}: c_v={sc:.3f} ({time.time()-t0:.1f}s)")
            if sc > best_score:
                best_k, best_score, best_model = k, sc, model

        print(f"      → best K={best_k} (c_v={best_score:.3f})")

        topics = [{"camp": camp, "topic_id": t,
                   "top_words": ", ".join(w for w, _ in
                                          best_model.show_topic(t, topn=10))}
                  for t in range(best_k)]
        for tp in topics:
            print(f"        T{tp['topic_id']}: {tp['top_words']}")

        results[camp] = {"model": best_model, "K": best_k,
                         "coherence": best_score, "topics": topics,
                         "scores": scores}

    return results


def plot_lda_topics(lda_results):
    if not lda_results:
        return
    n_camps = len(lda_results)
    fig, axes = plt.subplots(1, n_camps, figsize=(8 * n_camps, 9),
                              squeeze=False)

    for ax, (camp, res) in zip(axes.flat, lda_results.items()):
        model = res["model"]
        K = res["K"]
        rows = []
        for tid in range(K):
            for w, p in model.show_topic(tid, topn=8):
                rows.append((tid, w, p))

        y = 0
        topic_centres = []
        for tid in range(K):
            topic_words = [(w, p) for t, w, p in rows if t == tid]
            start = y
            for w, p in topic_words:
                ax.barh(y, p, color=PALETTE[camp], alpha=0.75,
                        edgecolor="white")
                ax.text(p + 0.001, y, f"  {w}", va="center", fontsize=9)
                y += 1
            topic_centres.append((start + y - 1) / 2)
            y += 0.6

        ax.set_yticks(topic_centres)
        ax.set_yticklabels([f"Topic {i}" for i in range(K)],
                            fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Word probability within topic")
        ax.set_title(f"{camp.title()} camp — LDA (K={K}, "
                     f"c_v={res['coherence']:.3f})", fontsize=13)
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("LDA — top words per topic, per camp", fontsize=15, y=1.00)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lda_topics.png", dpi=160,
                bbox_inches="tight")
    plt.close()


def plot_coherence(lda_results):
    if not lda_results:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for camp, res in lda_results.items():
        ks, scs = zip(*sorted(res["scores"].items()))
        ax.plot(ks, scs, marker="o", markersize=10, linewidth=2.5,
                color=PALETTE[camp], label=camp.title())
        ax.scatter([res["K"]], [res["coherence"]], s=300,
                   facecolor="none", edgecolor=PALETTE[camp], linewidth=2.5,
                   zorder=5)
        ax.annotate(f"chosen K = {res['K']}",
                    (res["K"], res["coherence"]),
                    xytext=(8, 12), textcoords="offset points",
                    color=PALETTE[camp], fontsize=10, fontweight="bold")
    ax.set_xlabel("Number of topics (K)")
    ax.set_ylabel("c_v coherence")
    ax.set_title("LDA coherence vs number of topics")
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lda_coherence.png", dpi=160,
                bbox_inches="tight")
    plt.close()


# FRAMING

def framing_analysis(df):
    print("\n  FRAMING (keyword vocabulary)")

    def count_kw(text, kws):
        text = (text or "").lower()
        return sum(text.count(k) for k in kws)

    df = df.copy()
    df["sci_kw"]  = df["body"].fillna("").apply(lambda t: count_kw(t, SCIENCE_KEYWORDS))
    df["den_kw"]  = df["body"].fillna("").apply(lambda t: count_kw(t, DENIAL_KEYWORDS))
    df["kw_lean"] = np.where(df["sci_kw"] > df["den_kw"], "science",
                     np.where(df["den_kw"] > df["sci_kw"], "denial", "neutral"))

    summary = (df.groupby("camp").agg(
                  sci_total=("sci_kw", "sum"),
                  denial_total=("den_kw", "sum"),
                  sci_per_comment=("sci_kw", "mean"),
                  denial_per_comment=("den_kw", "mean"),
                  pct_lean_science=("kw_lean", lambda s: (s == "science").mean()),
                  pct_lean_denial =("kw_lean", lambda s: (s == "denial").mean()))
                .reset_index())
    print(summary.round(3).to_string(index=False))
    return df, summary


def plot_framing(framing_df, summary_df):
    fig = plt.figure(figsize=(15, 9))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3)

    camps = summary_df["camp"].tolist()

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(camps))
    w = 0.35
    bars1 = ax.bar(x - w/2, summary_df["sci_per_comment"], w,
                   label="Science vocabulary", color=PALETTE["science"],
                   edgecolor="white")
    bars2 = ax.bar(x + w/2, summary_df["denial_per_comment"], w,
                   label="Denial vocabulary", color=PALETTE["denial"],
                   edgecolor="white")
    for bars in (bars1, bars2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([c.title() for c in camps])
    ax.set_ylabel("Mean keywords per comment")
    ax.set_title("Vocabulary intensity by camp")
    ax.legend(frameon=True, fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    ct = pd.crosstab(framing_df["camp"], framing_df["kw_lean"],
                      normalize="index")
    ct = ct.reindex(columns=["science", "neutral", "denial"]).fillna(0)
    bottoms = np.zeros(len(ct))
    lean_colors = {"science": PALETTE["science"],
                   "neutral": PALETTE["neutral"],
                   "denial":  PALETTE["denial"]}
    for col in ct.columns:
        ax.bar(ct.index, ct[col], bottom=bottoms,
               label=col.title(), color=lean_colors[col], edgecolor="white")
        for i, val in enumerate(ct[col]):
            if val > 0.05:
                ax.text(i, bottoms[i] + val/2, f"{val:.0%}",
                        ha="center", va="center", fontsize=10,
                        color="white", fontweight="bold")
        bottoms += ct[col].values
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion of comments")
    ax.set_title("Comment-level vocabulary lean within each camp")
    ax.legend(frameon=True, fontsize=10)

    ax = fig.add_subplot(gs[1, :])
    all_kw = sorted(SCIENCE_KEYWORDS | DENIAL_KEYWORDS)
    rows = []
    for camp in camps:
        bodies = framing_df[framing_df["camp"] == camp]["body"].fillna("").str.lower()
        for kw in all_kw:
            rows.append({"camp": camp, "keyword": kw,
                          "count": bodies.str.count(rf"\b{kw}\b").sum()})
    kw_df = pd.DataFrame(rows)
    matrix = kw_df.pivot(index="keyword", columns="camp", values="count")
    matrix_norm = matrix.div(matrix.sum(axis=0), axis=1) * 100
    matrix_norm = matrix_norm.fillna(0)
    top_kw = matrix.sum(axis=1).nlargest(20).index
    matrix_norm = matrix_norm.loc[top_kw]

    sns.heatmap(matrix_norm, ax=ax, cmap="Reds", annot=True, fmt=".2f",
                cbar_kws={"label": "% of camp's keyword hits"},
                linewidths=0.5, linecolor="white")
    ax.set_title("Top-20 keyword frequencies, normalised within each camp")
    ax.set_xlabel("")
    ax.set_ylabel("Keyword")

    plt.savefig(f"{OUTPUT_DIR}/framing_analysis.png",
                dpi=160, bbox_inches="tight")
    plt.close()


# CLASSIFIER

def classify_camps(df):
    print("\n  CLASSIFIER (TF-IDF + Logistic Regression)")

    counts = df["camp"].value_counts()
    minority = counts.min()
    balanced = (df.groupby("camp", group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), minority),
                                             random_state=42)))
    print(f"      balanced: {len(balanced):,} comments "
          f"({minority} per camp)")

    X = balanced["body"].fillna("").astype(str).values
    y = balanced["camp"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=42)

    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                          min_df=5, max_df=0.9,
                          stop_words="english", lowercase=True)
    Xtr_vec = vec.fit_transform(Xtr)
    Xte_vec = vec.transform(Xte)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=42)
    clf.fit(Xtr_vec, ytr)
    pred  = clf.predict(Xte_vec)
    proba = clf.predict_proba(Xte_vec)

    acc = accuracy_score(yte, pred)
    f1  = f1_score(yte, pred, average="macro")
    cm  = confusion_matrix(yte, pred, labels=clf.classes_)
    classes = list(clf.classes_)
    auc = roc_auc_score(yte == classes[1], proba[:, 1])

    print(f"      accuracy={acc:.3f}, macro-F1={f1:.3f}, ROC-AUC={auc:.3f}")
    print(classification_report(yte, pred, digits=3))

    features = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    top_pos = [(features[i], float(coefs[i]))
               for i in np.argsort(coefs)[-25:][::-1]]
    top_neg = [(features[i], float(coefs[i]))
               for i in np.argsort(coefs)[:25]]

    print(f"\n      top words → '{classes[1]}':")
    for w, c in top_pos[:8]:
        print(f"        {w:25s} {c:+.3f}")
    print(f"\n      top words → '{classes[0]}':")
    for w, c in top_neg[:8]:
        print(f"        {w:25s} {c:+.3f}")

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "roc_auc":  float(auc),
        "n_train": len(Xtr), "n_test": len(Xte),
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "y_test_proba": proba[:, 1].tolist(),
        "y_test_true":  (yte == classes[1]).astype(int).tolist(),
        f"top_{classes[1]}": top_pos,
        f"top_{classes[0]}": top_neg,
    }


def plot_classifier(metrics):
    classes = metrics["classes"]
    cm = np.array(metrics["confusion_matrix"])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    annot = np.array([[f"{c}\n({p:.1f}%)" for c, p in zip(row_c, row_p)]
                       for row_c, row_p in zip(cm, cm_pct)])
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=[c.title() for c in classes],
                yticklabels=[c.title() for c in classes],
                cbar_kws={"label": "Count"}, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    text = (f"PERFORMANCE\n\n"
            f"Accuracy   : {metrics['accuracy']:.3f}\n"
            f"Macro F1   : {metrics['macro_f1']:.3f}\n"
            f"ROC-AUC    : {metrics['roc_auc']:.3f}\n\n"
            f"Training   : {metrics['n_train']:,} comments\n"
            f"Test       : {metrics['n_test']:,} comments\n"
            f"Balanced   : Yes\n"
            f"Classes    : {', '.join(classes)}")
    ax.text(0.1, 0.5, text, fontsize=13, va="center",
            family="DejaVu Sans Mono",
            bbox=dict(facecolor="#F5F5F5", edgecolor="gray",
                      boxstyle="round,pad=1.0"))

    ax = fig.add_subplot(gs[0, 2])
    fpr, tpr, _ = roc_curve(metrics["y_test_true"], metrics["y_test_proba"])
    ax.plot(fpr, tpr, color="#1976D2", linewidth=2.5,
            label=f"ROC (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC — '{classes[1]}' vs rest")
    ax.legend(loc="lower right", frameon=True)

    pos = metrics[f"top_{classes[1]}"][:15]
    neg = metrics[f"top_{classes[0]}"][:15]

    ax = fig.add_subplot(gs[1, 0])
    words, coefs = zip(*pos)
    ax.barh(list(words)[::-1], list(coefs)[::-1],
            color=PALETTE.get(classes[1], "#1976D2"), edgecolor="white")
    ax.set_title(f"Top discriminative words → '{classes[1]}'")
    ax.set_xlabel("LR coefficient (+)")

    ax = fig.add_subplot(gs[1, 1])
    words, coefs = zip(*neg)
    ax.barh(list(words)[::-1], [abs(c) for c in list(coefs)[::-1]],
            color=PALETTE.get(classes[0], "#E53935"), edgecolor="white")
    ax.set_title(f"Top discriminative words → '{classes[0]}'")
    ax.set_xlabel("|LR coefficient|")

    ax = fig.add_subplot(gs[1, 2])
    proba = np.array(metrics["y_test_proba"])
    truth = np.array(metrics["y_test_true"])
    ax.hist(proba[truth == 1], bins=30, alpha=0.6,
             label=f"Actual = {classes[1]}", color=PALETTE.get(classes[1], "blue"))
    ax.hist(proba[truth == 0], bins=30, alpha=0.6,
             label=f"Actual = {classes[0]}", color=PALETTE.get(classes[0], "red"))
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"Predicted P({classes[1]})")
    ax.set_ylabel("Count")
    ax.set_title("Predicted-probability separation")
    ax.legend(frameon=True, fontsize=9)

    fig.suptitle("Misinformation classifier — TF-IDF + Logistic Regression",
                 fontsize=15, y=0.995)
    plt.savefig(f"{OUTPUT_DIR}/classifier_results.png",
                dpi=160, bbox_inches="tight")
    plt.close()


def main():

    print("STEP 4 — NLP ANALYSIS")


    path = f"{DATA_DIR}/clean_comments.csv"
    if not os.path.exists(path):
        sys.exit(f"  Missing {path}. Run 02_preprocessing.py first.")

    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} clean comments\n")

    df_sent = sentiment_analysis(df)
    keep_cols = [c for c in df_sent.columns if c != "tokens"]
    df_sent[keep_cols].to_csv(f"{DATA_DIR}/comments_sentiment.csv", index=False)
    plot_sentiment_dashboard(df_sent)

    lda_results = lda_per_camp(df_sent)
    if lda_results:
        topics_rows = [t for r in lda_results.values() for t in r["topics"]]
        pd.DataFrame(topics_rows).to_csv(f"{DATA_DIR}/topics.csv", index=False)
        plot_lda_topics(lda_results)
        plot_coherence(lda_results)

    framing_df, framing_summary = framing_analysis(df_sent)
    framing_summary.to_csv(f"{DATA_DIR}/framing.csv", index=False)
    plot_framing(framing_df, framing_summary)

    metrics = classify_camps(df_sent)
    with open(f"{DATA_DIR}/classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    plot_classifier(metrics)


    print(f"  Sentiment scored:    {len(df_sent):,} comments (RoBERTa)")
    print(f"  Topic models built:  {len(lda_results)} camps")
    print(f"  Classifier accuracy: {metrics['accuracy']:.3f}")
    print(f"  Classifier macro-F1: {metrics['macro_f1']:.3f}")
    print(f"  Classifier ROC-AUC:  {metrics['roc_auc']:.3f}")



if __name__ == "__main__":
    main()
