import os
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm import tqdm

from .utils import ensure_dir

ML25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def download_ml25m(data_dir: str) -> str:
    ensure_dir(data_dir)
    zip_path = os.path.join(data_dir, "ml-25m.zip")
    extract_dir = os.path.join(data_dir, "ml-25m")
    if os.path.isdir(extract_dir):
        return extract_dir
    if not os.path.exists(zip_path):
        import urllib.request

        print("Downloading MovieLens 25M...")
        urllib.request.urlretrieve(ML25M_URL, zip_path)
    print("Extracting MovieLens 25M...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)
    return extract_dir


def load_ratings(ratings_path: str) -> pd.DataFrame:
    df = pd.read_csv(ratings_path)
    df = df[["userId", "movieId", "rating", "timestamp"]]
    return df


def filter_users(df: pd.DataFrame, min_interactions: int) -> pd.DataFrame:
    if min_interactions <= 1:
        return df
    counts = df.groupby("userId").size()
    keep_users = counts[counts >= min_interactions].index
    return df[df["userId"].isin(keep_users)].copy()


def reindex(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    user_ids = np.sort(df["userId"].unique())
    item_ids = np.sort(df["movieId"].unique())
    user_map = {u: i for i, u in enumerate(user_ids)}
    item_map = {i: j for j, i in enumerate(item_ids)}
    df["userId"] = df["userId"].map(user_map)
    df["movieId"] = df["movieId"].map(item_map)
    return df, user_ids, item_ids


def leave_one_out_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["userId", "timestamp"])  # ascending
    train_rows = []
    val_rows = []
    test_rows = []
    for user_id, group in df.groupby("userId"):
        items = group.to_dict("records")
        if len(items) >= 3:
            train_rows.extend(items[:-2])
            val_rows.append(items[-2])
            test_rows.append(items[-1])
        elif len(items) == 2:
            train_rows.append(items[0])
            test_rows.append(items[1])
        else:
            train_rows.append(items[0])
    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows) if val_rows else pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(test_rows) if test_rows else pd.DataFrame(columns=df.columns)
    return train_df, val_df, test_df


def build_user_pos_list(
    num_users: int, train_df: pd.DataFrame
) -> List[np.ndarray]:
    user_pos = [list() for _ in range(num_users)]
    for row in train_df.itertuples(index=False):
        user_pos[row.userId].append(row.movieId)
    user_pos = [np.array(items, dtype=np.int64) for items in user_pos]
    return user_pos


def build_norm_adj(
    num_users: int, num_items: int, train_df: pd.DataFrame
) -> torch.sparse.FloatTensor:
    # Build user-item bipartite adjacency and normalize: D^-1/2 A D^-1/2
    rows = np.concatenate([train_df["userId"].values, train_df["movieId"].values + num_users])
    cols = np.concatenate([train_df["movieId"].values + num_users, train_df["userId"].values])
    data = np.ones(len(rows), dtype=np.float32)
    mat = sp.coo_matrix((data, (rows, cols)), shape=(num_users + num_items, num_users + num_items))
    deg = np.array(mat.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    d_mat = sp.diags(deg_inv_sqrt)
    norm_mat = d_mat @ mat @ d_mat
    norm_mat = norm_mat.tocoo()
    indices = torch.from_numpy(np.vstack([norm_mat.row, norm_mat.col]).astype(np.int64))
    values = torch.from_numpy(norm_mat.data.astype(np.float32))
    shape = torch.Size(norm_mat.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def save_mappings(data_dir: str, user_ids: np.ndarray, item_ids: np.ndarray) -> None:
    ensure_dir(data_dir)
    np.savez(os.path.join(data_dir, "mappings.npz"), user_ids=user_ids, item_ids=item_ids)


def load_mappings(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(os.path.join(data_dir, "mappings.npz"))
    return data["user_ids"], data["item_ids"]


def build_eval_data(num_users: int, df: pd.DataFrame) -> Dict[int, int]:
    eval_data = {}
    for row in df.itertuples(index=False):
        eval_data[row.userId] = row.movieId
    return eval_data


def prepare_data(data_dir: str, min_interactions: int) -> Dict[str, object]:
    extract_dir = download_ml25m(data_dir)
    ratings_path = os.path.join(extract_dir, "ratings.csv")
    df = load_ratings(ratings_path)
    df = filter_users(df, min_interactions)
    df, user_ids, item_ids = reindex(df)
    train_df, val_df, test_df = leave_one_out_split(df)

    num_users = df["userId"].nunique()
    num_items = df["movieId"].nunique()

    user_pos = build_user_pos_list(num_users, train_df)
    norm_adj = build_norm_adj(num_users, num_items, train_df)

    save_mappings(data_dir, user_ids, item_ids)

    return {
        "num_users": num_users,
        "num_items": num_items,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "user_pos": user_pos,
        "norm_adj": norm_adj,
        "user_ids": user_ids,
        "item_ids": item_ids,
    }


def sample_bpr_batch(
    user_pos: List[np.ndarray], num_items: int, batch_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    users = np.random.randint(0, len(user_pos), size=batch_size)
    pos_items = np.empty(batch_size, dtype=np.int64)
    neg_items = np.empty(batch_size, dtype=np.int64)
    for i, u in enumerate(users):
        pos = user_pos[u]
        if len(pos) == 0:
            pos_items[i] = np.random.randint(0, num_items)
        else:
            pos_items[i] = pos[np.random.randint(0, len(pos))]
        while True:
            neg = np.random.randint(0, num_items)
            if len(pos) == 0 or neg not in pos:
                neg_items[i] = neg
                break
    return users, pos_items, neg_items


def build_train_edge_index(train_df: pd.DataFrame) -> np.ndarray:
    # For potential debugging or analysis
    return train_df[["userId", "movieId"]].values
