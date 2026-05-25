import os
import sys
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import community as community_louvain


DATA_DIR   = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {"science": "#2196F3", "denial": "#F44336", "bridge": "#4CAF50"}

MIN_EDGE_WEIGHT    = 2
BETWEENNESS_SAMPLE = 300
TOP_N              = 20
NUM_PROP_SEEDS     = 10


def build_network(comments_df, camp):
    sub = comments_df[comments_df["camp"] == camp].copy()
    sub = sub.dropna(subset=["author", "parent_author"])
    sub = sub[sub["author"] != sub["parent_author"]]

    edges = (sub.groupby(["author", "parent_author"])
                .size().reset_index(name="weight"))
    edges = edges[edges["weight"] >= MIN_EDGE_WEIGHT]

    G = nx.DiGraph()
    for _, r in edges.iterrows():
        G.add_edge(r["author"], r["parent_author"], weight=int(r["weight"]))

    activity = sub["author"].value_counts().to_dict()
    nx.set_node_attributes(G, activity, "activity")
    nx.set_node_attributes(G, camp, "camp")

    print(f"  [{camp}] nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}")
    return G


def network_stats(G, camp):
    n, e = G.number_of_nodes(), G.number_of_edges()
    if n == 0:
        return {"camp": camp, "nodes": 0, "edges": 0}

    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))
    largest_cc = max(components, key=len)

    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    stats = {
        "camp": camp,
        "nodes": n,
        "edges": e,
        "density": nx.density(G),
        "avg_in_degree":  sum(in_deg.values())  / n,
        "avg_out_degree": sum(out_deg.values()) / n,
        "max_in_degree":  max(in_deg.values()),
        "max_out_degree": max(out_deg.values()),
        "num_components": len(components),
        "largest_cc_fraction": len(largest_cc) / n,
        "reciprocity":    nx.reciprocity(G) if e else 0.0,
        "avg_clustering": nx.average_clustering(undirected, weight="weight"),
    }

    if len(largest_cc) > 1:
        sub = undirected.subgraph(largest_cc)
        if len(largest_cc) > 500:
            sample = list(largest_cc)[:150]
            paths = []
            for u in sample:
                paths.extend(nx.single_source_shortest_path_length(sub, u).values())
            stats["avg_path_lcc"] = float(np.mean(paths)) if paths else 0.0
            stats["diameter_lcc"] = max(paths) if paths else 0
        else:
            stats["avg_path_lcc"] = nx.average_shortest_path_length(sub)
            stats["diameter_lcc"] = nx.diameter(sub)
    else:
        stats["avg_path_lcc"] = 0.0
        stats["diameter_lcc"] = 0

    return stats


def compute_centrality(G, camp):
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    print(f"  [{camp}] computing centrality")
    metrics = {}

    metrics["in_degree"]    = dict(G.in_degree(weight="weight"))
    metrics["out_degree"]   = dict(G.out_degree(weight="weight"))
    metrics["pagerank"]     = nx.pagerank(G, weight="weight", alpha=0.85)
    metrics["betweenness"]  = nx.betweenness_centrality(
        G, k=min(BETWEENNESS_SAMPLE, G.number_of_nodes()),
        weight="weight", seed=42)
    metrics["eigenvector"]  = nx.eigenvector_centrality_numpy(G, weight="weight")

    undirected = G.to_undirected()
    largest_cc = max(nx.connected_components(undirected), key=len)
    sub        = undirected.subgraph(largest_cc)
    cc_close   = nx.closeness_centrality(sub)
    metrics["closeness"] = {n: cc_close.get(n, 0.0) for n in G.nodes()}

    def norm(d):
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        return {k: 0.0 for k in d} if mx == mn else \
               {k: (v - mn) / (mx - mn) for k, v in d.items()}

    pr, bw, dg, cl = (norm(metrics[k])
                      for k in ("pagerank", "betweenness", "in_degree", "closeness"))
    metrics["influence_composite"] = {
        n: 0.40 * pr[n] + 0.30 * bw[n] + 0.20 * dg[n] + 0.10 * cl[n]
        for n in G.nodes()
    }

    df = pd.DataFrame(metrics)
    df["author"]   = df.index
    df["camp"]     = camp
    df["activity"] = df["author"].map(
        nx.get_node_attributes(G, "activity")).fillna(0).astype(int)
    return df.reset_index(drop=True)


def detect_communities(G, camp):
    if G.number_of_edges() == 0:
        return {}, 0.0
    undirected = G.to_undirected()
    partition  = community_louvain.best_partition(undirected, weight="weight",
                                                   random_state=42)
    modularity = community_louvain.modularity(partition, undirected,
                                               weight="weight")
    print(f"  [{camp}] {len(set(partition.values()))} communities, "
          f"modularity={modularity:.3f}")
    return partition, modularity


def krackhardt_ei(G, partition):
    E = I = 0
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        if partition.get(u) == partition.get(v):
            I += w
        else:
            E += w
    return 0.0 if (E + I) == 0 else (E - I) / (E + I)


def influence_propagation(G, seed_users, iterations=3):
    if G.number_of_edges() == 0:
        return {}

    out_w = defaultdict(float)
    for u, v, d in G.edges(data=True):
        out_w[u] += d.get("weight", 1)

    results = {}
    for seed in seed_users:
        if seed not in G:
            results[seed] = 0.0
            continue
        influence = defaultdict(float)
        influence[seed] = 1.0
        for _ in range(iterations):
            new_inf = defaultdict(float)
            for node, val in list(influence.items()):
                if val < 0.01:
                    continue
                for _, nbr, d in G.out_edges(node, data=True):
                    share = (d.get("weight", 1) / max(out_w[node], 1)) * val * 0.5
                    new_inf[nbr] += share
            for n, v in new_inf.items():
                influence[n] = max(influence[n], v)
        results[seed] = sum(influence.values())
    return results


def find_bridge_users(comments_df):
    print("  Finding cross-camp bridge users")
    user_camps = comments_df.groupby("author")["camp"].apply(set)
    bridges = user_camps[user_camps.apply(lambda s: len(s) >= 2)]

    rows = []
    for author in bridges.index:
        ur  = comments_df[comments_df["author"] == author]
        sci = (ur["camp"] == "science").sum()
        den = (ur["camp"] == "denial").sum()
        tot = sci + den
        sr  = sci / tot if tot else 0.0
        btype = ("science-leaning" if sr > 0.7
                 else "denial-leaning" if sr < 0.3
                 else "balanced")
        rows.append({"author": author,
                     "science_comments": int(sci),
                     "denial_comments":  int(den),
                     "total_comments":   int(tot),
                     "sci_ratio": sr,
                     "bridge_type": btype})

    df = pd.DataFrame(rows).sort_values("total_comments", ascending=False)
    print(f"  Found {len(df):,} bridge users")
    return df


def plot_network_stats(stats_sci, stats_den):
    metrics = ["density", "avg_in_degree", "avg_clustering", "reciprocity",
               "largest_cc_fraction"]
    labels  = ["Density", "Avg in-degree", "Clustering",
               "Reciprocity", "Largest CC %"]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w/2, [stats_sci.get(m, 0) for m in metrics], w,
           label="Science", color=PALETTE["science"])
    ax.bar(x + w/2, [stats_den.get(m, 0) for m in metrics], w,
           label="Denial",  color=PALETTE["denial"])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Network structural comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/network_stats_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_degree_distributions(G_sci, G_den):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, G, camp in [(axes[0], G_sci, "science"),
                         (axes[1], G_den, "denial")]:
        if G.number_of_nodes() == 0:
            continue
        ins  = [d for _, d in G.in_degree()  if d > 0]
        outs = [d for _, d in G.out_degree() if d > 0]
        ax.hist(ins,  bins=40, alpha=0.6, label="In-degree",
                color=PALETTE[camp], log=True)
        ax.hist(outs, bins=40, alpha=0.6, label="Out-degree",
                color="gray", log=True)
        ax.set_xlabel("Degree"); ax.set_ylabel("Count (log)")
        ax.set_title(f"{camp.title()} — degree distribution")
        ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/degree_distributions.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_centrality_comparison(cent_sci, cent_den, top_n=15):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, df, camp in [(axes[0], cent_sci, "science"),
                          (axes[1], cent_den, "denial")]:
        if df.empty:
            continue
        top = df.nlargest(top_n, "influence_composite")
        ax.barh(top["author"], top["influence_composite"],
                color=PALETTE[camp])
        ax.invert_yaxis()
        ax.set_xlabel("Composite influence score")
        ax.set_title(f"{camp.title()} — top {top_n} influential users")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/centrality_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_network_graph(G, partition, camp, max_nodes=300):
    if G.number_of_nodes() == 0:
        return
    undirected = G.to_undirected()
    largest_cc = max(nx.connected_components(undirected), key=len)
    sub        = undirected.subgraph(largest_cc)

    if len(sub) > max_nodes:
        top = sorted(sub.degree, key=lambda x: -x[1])[:max_nodes]
        sub = sub.subgraph([n for n, _ in top])

    pos     = nx.spring_layout(sub, k=0.5, iterations=50, seed=42)
    comms   = [partition.get(n, 0) for n in sub.nodes()]
    colours = cm.tab20(np.array(comms) / max(max(comms) + 1, 1))

    fig, ax = plt.subplots(figsize=(11, 11))
    nx.draw_networkx_edges(sub, pos, alpha=0.15, width=0.5, ax=ax)
    sizes = [max(15, sub.degree(n) * 8) for n in sub.nodes()]
    nx.draw_networkx_nodes(sub, pos, node_color=colours,
                           node_size=sizes, alpha=0.85, ax=ax)

    top5 = sorted(sub.degree, key=lambda x: -x[1])[:5]
    nx.draw_networkx_labels(sub, pos, labels={n: n for n, _ in top5},
                            font_size=9, ax=ax)
    ax.set_title(f"{camp.title()} network "
                 f"({len(sub):,} nodes, {sub.number_of_edges():,} edges) — "
                 f"coloured by community")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/network_graph_{camp}.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_community_analysis(stats_sci, stats_den):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    camps  = ["Science", "Denial"]
    colors = [PALETTE["science"], PALETTE["denial"]]

    for ax, key, title in [
        (axes[0], "num_communities", "Number of communities"),
        (axes[1], "modularity",      "Modularity (Louvain)"),
        (axes[2], "ei_index",        "E-I index (echo-chamber measure)"),
    ]:
        vals = [stats_sci.get(key, 0), stats_den.get(key, 0)]
        bars = ax.bar(camps, vals, color=colors)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v,
                    f"{v:.3f}" if isinstance(v, float) else f"{v}",
                    ha="center", va="bottom", fontsize=11)
        ax.set_title(title)
        if key == "ei_index":
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/community_analysis.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_bridge_users(bridge_df, top_n=25):
    if bridge_df.empty:
        return
    top = bridge_df.nlargest(top_n, "total_comments").iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(top["author"], top["science_comments"],
            color=PALETTE["science"], label="Science")
    ax.barh(top["author"], top["denial_comments"],
            left=top["science_comments"],
            color=PALETTE["denial"], label="Denial")
    ax.set_xlabel("Comments")
    ax.set_title(f"Top {top_n} cross-camp bridge users")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/bridge_users.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_influence_propagation(prop_sci, prop_den):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, prop, camp in [(axes[0], prop_sci, "science"),
                            (axes[1], prop_den, "denial")]:
        if not prop:
            continue
        items = sorted(prop.items(), key=lambda x: -x[1])
        names, vals = zip(*items)
        ax.bar(range(len(names)), vals, color=PALETTE[camp])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Total reach (cascade)")
        ax.set_title(f"{camp.title()} — influence propagation from top hubs")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/influence_propagation.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_centrality_scatter(cent_sci, cent_den):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, df, camp in [(axes[0], cent_sci, "science"),
                          (axes[1], cent_den, "denial")]:
        if df.empty:
            continue
        ax.scatter(df["pagerank"], df["betweenness"],
                   alpha=0.5, color=PALETTE[camp], s=18)
        top = df.nlargest(5, "influence_composite")
        for _, r in top.iterrows():
            ax.annotate(r["author"], (r["pagerank"], r["betweenness"]),
                        fontsize=8, alpha=0.85)
        ax.set_xlabel("PageRank")
        ax.set_ylabel("Betweenness centrality")
        ax.set_title(f"{camp.title()} — hubs vs brokers")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/centrality_scatter.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def main():
    t0 = time.time()
    
    print("STEP 3 — NETWORK ANALYSIS")


    path = f"{DATA_DIR}/clean_comments.csv"
    if not os.path.exists(path):
        sys.exit(f"  Missing {path}. Run 02_preprocessing.py first.")

    comments = pd.read_csv(path)
    print(f"  Loaded {len(comments):,} clean comments")
    print(f"  Camps: {dict(comments['camp'].value_counts())}\n")

    G_sci = build_network(comments, "science")
    G_den = build_network(comments, "denial")

    stats_sci = network_stats(G_sci, "science")
    stats_den = network_stats(G_den, "denial")

    cent_sci = compute_centrality(G_sci, "science")
    cent_den = compute_centrality(G_den, "denial")

    part_sci, mod_sci = detect_communities(G_sci, "science")
    part_den, mod_den = detect_communities(G_den, "denial")
    ei_sci = krackhardt_ei(G_sci, part_sci)
    ei_den = krackhardt_ei(G_den, part_den)
    print(f"  E-I index: science={ei_sci:+.3f}, denial={ei_den:+.3f}")

    stats_sci.update({"num_communities": len(set(part_sci.values())),
                      "modularity": mod_sci, "ei_index": ei_sci})
    stats_den.update({"num_communities": len(set(part_den.values())),
                      "modularity": mod_den, "ei_index": ei_den})

    top_sci_seeds = (cent_sci.nlargest(NUM_PROP_SEEDS, "influence_composite")
                     ["author"].tolist() if not cent_sci.empty else [])
    top_den_seeds = (cent_den.nlargest(NUM_PROP_SEEDS, "influence_composite")
                     ["author"].tolist() if not cent_den.empty else [])
    print("  Simulating influence propagation")
    prop_sci = influence_propagation(G_sci, top_sci_seeds)
    prop_den = influence_propagation(G_den, top_den_seeds)

    bridges = find_bridge_users(comments)

    pd.DataFrame([stats_sci, stats_den]).to_csv(
        f"{DATA_DIR}/network_stats.csv", index=False)
    pd.concat([cent_sci, cent_den], ignore_index=True).to_csv(
        f"{DATA_DIR}/centrality.csv", index=False)
    pd.concat([pd.DataFrame({"author": list(part_sci.keys()),
                              "community": list(part_sci.values()),
                              "camp": "science"}),
               pd.DataFrame({"author": list(part_den.keys()),
                              "community": list(part_den.values()),
                              "camp": "denial"})],
              ignore_index=True).to_csv(
        f"{DATA_DIR}/communities.csv", index=False)
    bridges.to_csv(f"{DATA_DIR}/bridge_users.csv", index=False)
    pd.DataFrame([{"camp": camp, "author": u, "reach": r}
                  for camp, prop in [("science", prop_sci),
                                       ("denial",  prop_den)]
                  for u, r in prop.items()]).to_csv(
        f"{DATA_DIR}/influence_propagation.csv", index=False)

    print("\n  Generating figures")
    plot_network_stats(stats_sci, stats_den)
    plot_degree_distributions(G_sci, G_den)
    plot_centrality_comparison(cent_sci, cent_den)
    plot_network_graph(G_sci, part_sci, "science")
    plot_network_graph(G_den, part_den, "denial")
    plot_community_analysis(stats_sci, stats_den)
    plot_bridge_users(bridges)
    plot_influence_propagation(prop_sci, prop_den)
    plot_centrality_scatter(cent_sci, cent_den)


    summary = pd.DataFrame([stats_sci, stats_den])
    cols = ["camp", "nodes", "edges", "density", "avg_clustering",
            "num_communities", "modularity", "ei_index"]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].to_string(index=False))
    print(f"\n  Bridge users: {len(bridges):,}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")



if __name__ == "__main__":
    main()
