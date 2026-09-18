"""
Phase 3 — Latent Space Study
Projects the 128-dim Musical Narrative Vectors into 2D via UMAP to test
the hypothesis: Human tracks scatter widely (high structural rarity),
while AI tracks clump into a tight, dense cluster ("AI convergence").

Usage:
    python umap_analysis.py --narrative narrative_vectors.parquet --labels_col label
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import umap


def plot_umap(df: pd.DataFrame, feature_cols: list, label_col: str, out_path: str):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(df[feature_cols].values)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, name, color in [(0, "Human", "#2b6cb0"), (1, "AI-generated", "#e53e3e")]:
        mask = df[label_col] == label
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=12, alpha=0.6, label=name, c=color,
        )
    ax.set_title("UMAP of Musical Narrative Vectors")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--narrative", required=True, help="Parquet of narrative vectors")
    parser.add_argument("--labels_col", default="label")
    parser.add_argument("--out", default="umap_narrative.png")
    args = parser.parse_args()

    df = pd.read_parquet(args.narrative)
    feature_cols = [c for c in df.columns if c not in ("path", args.labels_col)]
    plot_umap(df, feature_cols, args.labels_col, args.out)


if __name__ == "__main__":
    main()
