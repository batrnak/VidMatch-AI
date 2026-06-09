import os
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    config_path = "Model/lightgcn-1M/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = "checkpoints/lightgcn.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    state_dict = ckpt["model_state"]
    item_emb_weight = state_dict["item_embedding.weight"].cpu().numpy()
    print("Item embeddings shape:", item_emb_weight.shape)

    mappings_path = os.path.join(cfg["data_dir"], "mappings.npz")
    if not os.path.exists(mappings_path):
        print(f"Mappings not found at {mappings_path}")
        return
    mappings = np.load(mappings_path)
    item_ids = mappings["item_ids"]

    movies_path = "datasets/ml-1m/movies.dat"
    if not os.path.exists(movies_path):
        print(f"Movies metadata not found at {movies_path}")
        return
    
    m_cols = ["movieId", "title", "genres"]
    movies_df = pd.read_csv(movies_path, sep="::", names=m_cols, encoding="latin-1", engine="python")
    
    genre_list = []
    title_list = []
    valid_indices = []
    
    movie_dict = movies_df.set_index("movieId").to_dict(orient="index")
    
    for idx, raw_id in enumerate(item_ids):
        if raw_id in movie_dict:
            title = movie_dict[raw_id]["title"]
            genres = movie_dict[raw_id]["genres"].split("|")
            main_genre = genres[0]
            genre_list.append(main_genre)
            title_list.append(title)
            valid_indices.append(idx)
            
    filtered_embeddings = item_emb_weight[valid_indices]
    
    print(f"Projecting {len(filtered_embeddings)} movies using t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(filtered_embeddings)
    
    plt.figure(figsize=(14, 10), dpi=150)
    
    plot_df = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "Genre": genre_list,
        "Title": title_list
    })
    
    top_genres = plot_df["Genre"].value_counts().nlargest(10).index.tolist()
    plot_df["Genre_Cleaned"] = plot_df["Genre"].apply(lambda g: g if g in top_genres else "Other")
    
    sns.scatterplot(
        x="x", y="y",
        hue="Genre_Cleaned",
        palette="tab20",
        data=plot_df,
        alpha=0.6,
        edgecolor=None,
        s=15
    )
    
    famous_movies = [
        "Toy Story (1995)",
        "Jurassic Park (1993)",
        "Star Wars: Episode IV - A New Hope (1977)",
        "Terminator 2: Judgment Day (1991)",
        "Godfather, The (1972)",
        "Matrix, The (1999)",
        "Lion King, The (1994)",
        "Silence of the Lambs, The (1991)",
        "Pulp Fiction (1994)",
        "Forrest Gump (1994)"
    ]
    
    for movie in famous_movies:
        match = plot_df[plot_df["Title"].str.contains(movie, case=False, na=False, regex=False)]
        if not match.empty:
            row = match.iloc[0]
            plt.annotate(
                row["Title"],
                xy=(row["x"], row["y"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                weight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.5)
            )
            
    plt.title("LightGCN Movie Embeddings Visualization (t-SNE on MovieLens 1M)", fontsize=16, weight="bold")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Main Genre", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    
    output_plot_path = "checkpoints/embeddings_visualization.png"
    plt.savefig(output_plot_path, bbox_inches="tight")
    print(f"Visualization saved to: {output_plot_path}")

if __name__ == '__main__':
    main()
