import os
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

PALETTE = {"science": "#2196F3", "denial": "#F44336", "bridge": "#4CAF50",
           "neutral": "#9E9E9E"}


def load_all():
    needed = {
        "comments":    f"{DATA_DIR}/comments_sentiment.csv",
        "centrality":  f"{DATA_DIR}/centrality.csv",
        "communities": f"{DATA_DIR}/communities.csv",
        "bridges":     f"{DATA_DIR}/bridge_users.csv",
        "stats":       f"{DATA_DIR}/network_stats.csv",
    }
    for k, p in needed.items():
        if not os.path.exists(p):
            sys.exit(f"  Missing {p}. Run earlier steps first.")
    return {k: pd.read_csv(p) for k, p in needed.items()}


def per_user_sentiment(comments):
    return (comments.groupby(["author", "camp"])
                    .agg(mean_compound=("sent_compound", "mean"),
                         std_compound=("sent_compound", "std"),
                         n_comments=("sent_compound", "size"),
                         pct_negative=("sent_label",
                                        lambda s: (s == "negative").mean()),
                         pct_positive=("sent_label",
                                        lambda s: (s == "positive").mean()))
                    .reset_index())


def centrality_vs_sentiment(centrality, user_sent):
    merged = centrality.merge(
        user_sent[["author", "camp", "mean_compound",
                   "pct_negative", "n_comments"]],
        on=["author", "camp"], how="left"
    )
    merged["mean_compound"] = merged["mean_compound"].fillna(0)
    return merged


def community_sentiment(communities, user_sent):
    m = communities.merge(user_sent, on=["author", "camp"], how="left")
    return (m.groupby(["camp", "community"])
              .agg(community_size=("author", "size"),
                   mean_compound=("mean_compound", "mean"),
                   pct_negative=("pct_negative", "mean"),
                   active_users=("n_comments",
                                  lambda s: (s.fillna(0) > 0).sum()))
              .reset_index()
              .sort_values(["camp", "community_size"],
                            ascending=[True, False]))


def bridge_sentiment(bridges, user_sent):
    sent_any = (user_sent.groupby("author")
                          .agg(mean_compound=("mean_compound", "mean"),
                               n_comments=("n_comments", "sum"))
                          .reset_index())
    return bridges.merge(sent_any, on="author", how="left")


def plot_centrality_vs_sentiment(merged):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, camp in zip(axes, ["science", "denial"]):
        sub = merged[merged["camp"] == camp]
        if sub.empty:
            continue
        sub = sub[sub["n_comments"].fillna(0) >= 2]
        ax.scatter(sub["influence_composite"], sub["mean_compound"],
                   alpha=0.45, s=18, color=PALETTE[camp])
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xlabel("Composite influence score")
        ax.set_ylabel("Mean sentiment compound")
        ax.set_title(f"{camp.title()} — influence vs sentiment (n={len(sub):,})")
        if len(sub) > 5:
            corr = sub[["influence_composite", "mean_compound"]].corr().iloc[0, 1]
            ax.text(0.05, 0.95, f"r = {corr:+.3f}", transform=ax.transAxes,
                    fontsize=11, va="top",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/centrality_vs_sentiment.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_community_sentiment(profiles):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, camp in zip(axes, ["science", "denial"]):
        sub = profiles[profiles["camp"] == camp]
        if sub.empty:
            continue
        sizes = sub["community_size"]
        ax.scatter(sizes, sub["mean_compound"],
                   s=np.sqrt(sizes) * 8, alpha=0.65, color=PALETTE[camp])
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xlabel("Community size")
        ax.set_ylabel("Mean sentiment compound")
        ax.set_title(f"{camp.title()} — community size × sentiment")
        ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/community_sentiment.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_bridge_sentiment(bridge_sent):
    if bridge_sent.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    types = ["science-leaning", "balanced", "denial-leaning"]
    data = [bridge_sent[bridge_sent["bridge_type"] == t]["mean_compound"]
              .dropna().values for t in types]
    bp = axes[0].boxplot(data, labels=types, patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"],
                         [PALETTE["science"], PALETTE["bridge"],
                          PALETTE["denial"]]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Mean sentiment compound")
    axes[0].set_title("Bridge-user sentiment by lean")

    counts = bridge_sent["bridge_type"].value_counts()
    axes[1].bar(counts.index, counts.values,
                color=[PALETTE["science"], PALETTE["bridge"],
                       PALETTE["denial"]][:len(counts)])
    for i, v in enumerate(counts.values):
        axes[1].text(i, v, f"{v}", ha="center", va="bottom")
    axes[1].set_title("Bridge-user count by lean")
    axes[1].set_ylabel("Users")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/bridge_sentiment.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_final_summary(stats, merged, profiles, bridges):
    fig = plt.figure(figsize=(15, 11))
    gs  = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(stats["camp"], stats["nodes"],
           color=[PALETTE.get(c, "gray") for c in stats["camp"]])
    for i, v in enumerate(stats["nodes"]):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom")
    ax.set_title("Network size (nodes)")
    ax.set_ylabel("Users")

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(stats))
    ax.bar(x - 0.2, stats["modularity"], 0.4,
           label="Modularity", color="#1976D2")
    ax.bar(x + 0.2, stats["ei_index"], 0.4,
           label="E-I index", color="#E64A19")
    ax.set_xticks(x); ax.set_xticklabels(stats["camp"])
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("Community structure: modularity vs E-I")
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    sent_by_camp = (merged.groupby("camp")["mean_compound"].mean()
                          .reindex(stats["camp"]).fillna(0))
    ax.bar(sent_by_camp.index, sent_by_camp.values,
           color=[PALETTE.get(c, "gray") for c in sent_by_camp.index])
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("Mean per-user sentiment")
    ax.set_ylabel("Compound score")

    ax = fig.add_subplot(gs[1, :])
    for camp in ["science", "denial"]:
        sub = merged[(merged["camp"] == camp) &
                      (merged["n_comments"].fillna(0) >= 2)]
        if sub.empty: continue
        ax.scatter(sub["influence_composite"], sub["mean_compound"],
                   alpha=0.4, s=15, color=PALETTE[camp], label=camp)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Composite influence score")
    ax.set_ylabel("Mean user sentiment")
    ax.set_title("Influence vs sentiment across both camps")
    ax.legend()

    ax = fig.add_subplot(gs[2, 0])
    if not bridges.empty:
        counts = bridges["bridge_type"].value_counts()
        ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%",
               colors=[PALETTE.get("science"), PALETTE.get("bridge"),
                       PALETTE.get("denial")][:len(counts)])
    ax.set_title(f"Bridge users: {len(bridges)}")

    ax = fig.add_subplot(gs[2, 1:])
    rows = []
    for camp in ["science", "denial"]:
        sub = merged[merged["camp"] == camp]
        if sub.empty: continue
        top = sub.nlargest(5, "influence_composite")
        for _, r in top.iterrows():
            rows.append({"author": r["author"], "camp": camp,
                          "score": r["influence_composite"]})
    if rows:
        d = pd.DataFrame(rows).iloc[::-1]
        ax.barh(d["author"], d["score"],
                color=[PALETTE[c] for c in d["camp"]])
        ax.set_xlabel("Composite influence")
        ax.set_title("Top 5 influencers per camp")

    plt.suptitle("Climate Misinformation vs Scientific Consensus — "
                 "Executive Summary", fontsize=14, y=0.995)
    plt.savefig(f"{OUTPUT_DIR}/final_summary.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def build_summary_table(stats, bridges):
    rows = []
    for _, r in stats.iterrows():
        rows.append({
            "Camp": r["camp"].title(),
            "Users (nodes)":  int(r["nodes"]),
            "Interactions (edges)": int(r["edges"]),
            "Density":      round(float(r["density"]), 5),
            "Clustering":   round(float(r["avg_clustering"]), 4),
            "Communities":  int(r.get("num_communities", 0)),
            "Modularity":   round(float(r.get("modularity", 0)), 3),
            "E-I index":    round(float(r.get("ei_index", 0)), 3),
        })
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"Camp": "Bridge users",
                       "Users (nodes)": len(bridges),
                       "Interactions (edges)": "",
                       "Density": "", "Clustering": "",
                       "Communities": "", "Modularity": "",
                       "E-I index": ""}
    df.to_csv(f"{DATA_DIR}/results_summary.csv", index=False)
    return df


def main():

    print("STEP 5 — INTEGRATED RESULTS")


    artefacts = load_all()
    comments    = artefacts["comments"]
    centrality  = artefacts["centrality"]
    communities = artefacts["communities"]
    bridges     = artefacts["bridges"]
    stats       = artefacts["stats"]

    print(f"  Comments: {len(comments):,}, "
          f"users: {centrality['author'].nunique():,}, "
          f"bridges: {len(bridges)}")

    user_sent = per_user_sentiment(comments)
    user_sent.to_csv(f"{DATA_DIR}/user_sentiment.csv", index=False)

    merged = centrality_vs_sentiment(centrality, user_sent)
    merged.to_csv(f"{DATA_DIR}/centrality_sentiment.csv", index=False)

    profiles = community_sentiment(communities, user_sent)
    profiles.to_csv(f"{DATA_DIR}/community_sentiment.csv", index=False)

    brsent = bridge_sentiment(bridges, user_sent)
    brsent.to_csv(f"{DATA_DIR}/bridge_sentiment.csv", index=False)

    print("\n  Generating figures")
    plot_centrality_vs_sentiment(merged)
    plot_community_sentiment(profiles)
    plot_bridge_sentiment(brsent)
    plot_final_summary(stats, merged, profiles, bridges)

    table = build_summary_table(stats, bridges)


    print(table.to_string(index=False))
    
    
    print("\n  Sucessfully generated integrated results and summary table.")


if __name__ == "__main__":
    main()
