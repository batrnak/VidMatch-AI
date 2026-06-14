import os
import json
import pandas as pd
import numpy as np

def filter_users(df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    if min_interactions <= 1:
        return df
    counts = df.groupby("userId").size()
    keep_users = counts[counts >= min_interactions].index
    return df[df["userId"].isin(keep_users)].copy()

def reindex(df: pd.DataFrame):
    user_ids = np.sort(df["userId"].unique())
    item_ids = np.sort(df["movieId"].unique())
    user_map = {u: i for i, u in enumerate(user_ids)}
    item_map = {i: j for j, i in enumerate(item_ids)}
    df["userId"] = df["userId"].map(user_map)
    df["movieId"] = df["movieId"].map(item_map)
    return df, len(user_ids), len(item_ids)

def leave_one_out_split(df: pd.DataFrame):
    df = df.sort_values(["userId", "timestamp"])  # ascending
    train_rows = []
    val_rows = []
    test_rows = []
    for user_id, group in df.groupby("userId"):
        items = group.to_dict("records")
        if len(items) >= 3:
            test_rows.append(items.pop())
            val_rows.append(items.pop())
            train_rows.extend(items)
        else:
            train_rows.extend(items)

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    return train_df, val_df, test_df

def main():
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(proj_root, "datasets", "ml-1m")
    raw_path = os.path.join(data_dir, "ratings.dat")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(
        raw_path,
        sep="::",
        engine="python",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
    )

    print("Filtering users with < 10 interactions...")
    df = filter_users(df, min_interactions=10)

    print("Reindexing users and items to 0..N-1...")
    df, num_users, num_items = reindex(df)

    print("Splitting data (Leave-One-Out)...")
    train_df, val_df, test_df = leave_one_out_split(df)

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    print("Saving to CSV files...")
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    metadata = {
        "num_users": num_users,
        "num_items": num_items,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df)
    }
    with open(os.path.join(processed_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("Data processing and saving completed successfully!")

if __name__ == "__main__":
    main()
