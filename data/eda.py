import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    ratings_path = "datasets/ml-1m/ratings.dat"
    if not os.path.exists(ratings_path):
        print(f"Dataset not found at {ratings_path}. Please run training first to download it.")
        return

    print("Loading ratings data...")
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(ratings_path, sep='::', names=r_cols, encoding='latin-1', engine='python')

    # Calculate statistics
    num_users = ratings["user_id"].nunique()
    num_items = ratings["movie_id"].nunique()
    num_ratings = len(ratings)
    sparsity = 1 - num_ratings / (num_users * num_items)

    print("\n=== DATASET STATISTICS ===")
    print(f"Users: {num_users}")
    print(f"Items: {num_items}")
    print(f"Ratings: {num_ratings}")
    print(f"Sparsity (Độ thưa thớt): {sparsity:.4%}")
    print("==========================\n")

    # Ensure checkpoints directory exists
    os.makedirs("checkpoints", exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Rating Distribution Plot
    print("Generating rating distribution plot...")
    plt.figure(figsize=(8, 5), dpi=150)
    ratings["rating"].value_counts().sort_index().plot(kind="bar", color="#34495e", edgecolor="black")
    plt.title("Phân Phối Điểm Đánh Giá (Rating Distribution)", fontsize=14, weight='bold')
    plt.xlabel("Rating (Số sao)", fontsize=12)
    plt.ylabel("Count (Số lượng)", fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("checkpoints/eda_rating_distribution.png")
    plt.close()

    # 2. Interactions per User Plot
    print("Generating interactions per user plot...")
    user_counts = ratings.groupby("user_id").size()
    plt.figure(figsize=(8, 5), dpi=150)
    user_counts.plot(kind="hist", bins=50, color="#2980b9", edgecolor="black")
    plt.title("Số Lượt Tương Tác Của Mỗi Người Dùng (Interactions per User)", fontsize=14, weight='bold')
    plt.xlabel("Số lượng đánh giá", fontsize=12)
    plt.ylabel("Số lượng User", fontsize=12)
    plt.tight_layout()
    plt.savefig("checkpoints/eda_user_interactions.png")
    plt.close()

    # 3. Interactions per Item Plot
    print("Generating interactions per item plot...")
    item_counts = ratings.groupby("movie_id").size()
    plt.figure(figsize=(8, 5), dpi=150)
    item_counts.plot(kind="hist", bins=50, color="#e67e22", edgecolor="black")
    plt.title("Số Lượt Tương Tác Của Mỗi Bộ Phim (Interactions per Item)", fontsize=14, weight='bold')
    plt.xlabel("Số lượng đánh giá", fontsize=12)
    plt.ylabel("Số lượng Phim", fontsize=12)
    plt.tight_layout()
    plt.savefig("checkpoints/eda_item_interactions.png")
    plt.close()

    # 4. Ratings Over Time Plot
    print("Generating ratings over time plot...")
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["year_month"] = ratings["datetime"].dt.to_period("M").astype(str)
    monthly_counts = ratings.groupby("year_month").size()

    plt.figure(figsize=(12, 5), dpi=150)
    monthly_counts.plot(color="#27ae60", linewidth=2.5)
    plt.title("Xu Hướng Số Lượng Đánh Giá Theo Thời Gian (Ratings Over Time)", fontsize=14, weight='bold')
    plt.xlabel("Thời gian (Năm-Tháng)", fontsize=12)
    plt.ylabel("Số lượng đánh giá", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("checkpoints/eda_ratings_over_time.png")
    plt.close()

    print("All EDA plots generated and saved successfully under 'checkpoints/'.")

if __name__ == '__main__':
    main()
