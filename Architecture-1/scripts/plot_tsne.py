"""
t-SNE 2D Manifold Projection of Musical Latent Space
Projects 128-D Narrative & Fused representations onto 2D space across
Human and 6 AI Generator families.
"""
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def main():
    print("[1/3] Loading narrative vectors and generator metadata...")
    t0 = time.time()
    meta_path = Path("d:/mirex/data/processed/train_val_split_200k.parquet")
    narr_path = Path("d:/mirex/data/processed/narrative_vectors_200k_train.parquet")
    
    df_meta = pd.read_parquet(meta_path)[["file_path", "generator_family", "is_ai"]]
    df_narr = pd.read_parquet(narr_path)
    
    merged = df_narr.merge(df_meta, left_on="path", right_on="file_path", how="inner")
    
    # Feature columns (numerical narrative dimensions)
    feat_cols = [c for c in df_narr.columns if c not in ["path", "file_path", "generator_family", "is_ai", "strat_class", "source_dataset", "split"]]
    feat_cols = [c for c in feat_cols if np.issubdtype(df_narr[c].dtype, np.number)]
    print(f"      Loaded {len(merged):,d} tracks with {len(feat_cols)} narrative dimensions in {time.time()-t0:.1f}s")

    # Sample 600 tracks per generator for clean, responsive t-SNE projection (~4,200 points)
    print("[2/3] Sampling balanced subset (600 tracks per generator)...")
    sample_subsets = []
    for gen, group in merged.groupby("generator_family"):
        n = min(len(group), 600)
        sample_subsets.append(group.sample(n=n, random_state=42))
    df_sample = pd.concat(sample_subsets).reset_index(drop=True)

    X = df_sample[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=50.0, neginf=-50.0)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    print(f"[3/3] Running t-SNE on {len(X_norm):,d} tracks...")
    t1 = time.time()
    tsne = TSNE(
        n_components=2,
        perplexity=35,
        early_exaggeration=12.0,
        learning_rate="auto",
        init="pca",
        max_iter=1000,
        random_state=42,
        n_jobs=8,
    )
    X_2d = tsne.fit_transform(X_norm)
    print(f"      t-SNE converged in {time.time()-t1:.1f}s")

    df_sample["tsne_x"] = X_2d[:, 0]
    df_sample["tsne_y"] = X_2d[:, 1]

    # Plot styling
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#161922")
    ax.grid(True, linestyle="--", alpha=0.15, color="#ffffff")

    palette = {
        "human": "#e07a5f",       # Warm terracotta/orange
        "suno": "#3d5a80",        # Deep blue
        "udio": "#e63946",        # Vivid red
        "echoes": "#f4a261",      # Sandy gold
        "musicgen": "#9b5de5",    # Purple
        "audioldm": "#00f5d4",    # Cyan / mint
        "mustango": "#fee440",    # Bright yellow
    }

    labels_map = {
        "human": "Human Music (Studio/FMA)",
        "suno": "Suno v5 (Commercial AI)",
        "udio": "Udio (Diffusion AI)",
        "echoes": "StableAudio / Echoes",
        "musicgen": "MusicGen (Autoregressive)",
        "audioldm": "AudioLDM (Diffusion)",
        "mustango": "Mustango (Text-to-Music)",
    }

    # Plot each generator family
    for gen in ["suno", "udio", "mustango", "audioldm", "musicgen", "echoes", "human"]:
        sub = df_sample[df_sample["generator_family"] == gen]
        if len(sub) == 0:
            continue
        col = palette.get(gen, "#888888")
        alpha = 0.85 if gen == "human" else 0.70
        size = 36 if gen == "human" else 28
        ax.scatter(
            sub["tsne_x"],
            sub["tsne_y"],
            c=col,
            label=labels_map.get(gen, gen),
            alpha=alpha,
            s=size,
            edgecolors="none",
        )

    ax.set_title("t-SNE 2D Manifold: Musical Narrative Latent Space (MusicScope-CL)", fontsize=14, color="white", pad=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1 (Non-linear manifold axis)", fontsize=11, color="#bbbbbb")
    ax.set_ylabel("t-SNE Dimension 2 (Non-linear manifold axis)", fontsize=11, color="#bbbbbb")
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_color("#333842")

    leg = ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="#1c202a",
        edgecolor="#333842",
        fontsize=9.5,
    )
    for text in leg.get_texts():
        text.set_color("white")

    plt.tight_layout()
    out_img = Path("d:/mirex/Architecture-1/scripts/tsne_latent_space.png")
    plt.savefig(out_img, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Done] t-SNE plot saved to: {out_img}")

if __name__ == "__main__":
    main()
