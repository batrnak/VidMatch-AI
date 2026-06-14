import os
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm import tqdm

try:
    from .utils import ensure_dir
except ImportError:
    from utils import ensure_dir

import json

def load_processed_data(data_dir: str):
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    processed_dir = os.path.join(proj_root, "datasets", "ml-1m", "processed")
    
    with open(os.path.join(processed_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
        
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    
    return train_df, val_df, test_df, metadata["num_users"], metadata["num_items"]


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


def prepare_data(data_dir: str, min_interactions: int = 0) -> Dict[str, object]:
    train_df, val_df, test_df, num_users, num_items = load_processed_data(data_dir)
    
    user_pos = build_user_pos_list(num_users, train_df)
    norm_adj = build_norm_adj(num_users, num_items, train_df)
    
    # We no longer save mappings here because it's centralized in the generation script.

    user_pos_set = [set(pos.tolist()) for pos in user_pos]

    return {
        "num_users": num_users,
        "num_items": num_items,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "user_pos": user_pos,
        "user_pos_set": user_pos_set,
        "norm_adj": norm_adj,
        "user_ids": np.arange(num_users),
        "item_ids": np.arange(num_items),
    }


def sample_bpr_batch(
    user_pos: List[np.ndarray], user_pos_set: List[set], num_items: int, batch_size: int
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
        u_set = user_pos_set[u]
        while True:
            neg = np.random.randint(0, num_items)
            if len(u_set) == 0 or neg not in u_set:
                neg_items[i] = neg
                break
    return users, pos_items, neg_items


def build_train_edge_index(train_df: pd.DataFrame) -> np.ndarray:
    # For potential debugging or analysis
    return train_df[["userId", "movieId"]].values
