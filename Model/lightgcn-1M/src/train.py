import argparse
import os
from typing import Dict

import matplotlib
import numpy as np
import torch
import yaml
from tqdm import tqdm

try:
    from .data import prepare_data, sample_bpr_batch, build_eval_data
    from .metrics import recall_at_k, ndcg_at_k
    from .model import LightGCN
    from .utils import ensure_dir, get_device, set_seed
except ImportError:
    from data import prepare_data, sample_bpr_batch, build_eval_data
    from metrics import recall_at_k, ndcg_at_k
    from model import LightGCN
    from utils import ensure_dir, get_device, set_seed

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def evaluate(
    model: LightGCN,
    norm_adj: torch.sparse.FloatTensor,
    eval_data: Dict[int, int],
    user_pos,
    num_items: int,
    k: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.propagate(norm_adj)
        recalls = []
        ndcgs = []
        users = np.array(list(eval_data.keys()), dtype=np.int64)
        for start in tqdm(range(0, len(users), batch_size), desc="eval", leave=False):
            end = min(start + batch_size, len(users))
            batch_users = users[start:end]
            batch_users_t = torch.from_numpy(batch_users).to(device)
            scores = model.full_sort_scores(user_emb, item_emb, batch_users_t).cpu().numpy()
            for i, u in enumerate(batch_users):
                seen = user_pos[u]
                if len(seen) > 0:
                    scores[i, seen] = -np.inf
            topk_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            topk_scores = np.take_along_axis(scores, topk_idx, axis=1)
            topk_order = np.argsort(-topk_scores, axis=1)
            ranklist = np.take_along_axis(topk_idx, topk_order, axis=1)
            for i, u in enumerate(batch_users):
                gt = eval_data[u]
                recalls.append(recall_at_k(ranklist[i], gt, k))
                ndcgs.append(ndcg_at_k(ranklist[i], gt, k))
        return {"recall": float(np.mean(recalls)), "ndcg": float(np.mean(ndcgs))}


def plot_training_metrics(
    recalls: list[float], ndcgs: list[float], k: int, output_path: str
) -> None:
    epochs = np.arange(1, len(recalls) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)

    axes[0].plot(epochs, recalls, marker="o", linewidth=1.8)
    axes[0].set_title(f"Recall@{k} on Test")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(f"Recall@{k}")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, ndcgs, marker="o", linewidth=1.8, color="tab:orange")
    axes[1].set_title(f"NDCG@{k} on Test")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(f"NDCG@{k}")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = args.config
    if not os.path.exists(config_path):
        fallback_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_path)
        if os.path.exists(fallback_path):
            config_path = fallback_path

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["seed"]))
    data_dir = cfg["data_dir"]
    ensure_dir(cfg["checkpoint_dir"])

    data = prepare_data(data_dir, int(cfg["min_interactions"]))
    device = get_device()

    model = LightGCN(
        data["num_users"],
        data["num_items"],
        int(cfg["embedding_dim"]),
        int(cfg["num_layers"]),
    ).to(device)

    norm_adj = data["norm_adj"].to(device)
    user_pos = data["user_pos"]

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))

    eval_data = build_eval_data(data["num_users"], data["test_df"])

    steps_per_epoch = max(1, len(data["train_df"]) // int(cfg["batch_size"]))
    recall_history = []
    ndcg_history = []
    eval_k = int(cfg["eval_k"])
    last_metrics = {"recall": 0.0, "ndcg": 0.0}
    eval_every = 10

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch}")
        for _ in pbar:
            users, pos_items, neg_items = sample_bpr_batch(
                user_pos, data["user_pos_set"], data["num_items"], int(cfg["batch_size"])
            )
            users_t = torch.from_numpy(users).to(device)
            pos_t = torch.from_numpy(pos_items).to(device)
            neg_t = torch.from_numpy(neg_items).to(device)

            user_emb, item_emb = model.propagate(norm_adj)
            loss = model.bpr_loss(
                users_t,
                pos_t,
                neg_t,
                user_emb,
                item_emb,
                float(cfg["reg_weight"]),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": float(loss.item())})

        if epoch == 1 or epoch % eval_every == 0 or epoch == int(cfg["epochs"]):
            metrics = evaluate(
                model,
                norm_adj,
                eval_data,
                user_pos,
                data["num_items"],
                eval_k,
                int(cfg["eval_batch_size"]),
                device,
            )
            last_metrics = metrics
            print(f"Epoch {epoch}: Recall@{eval_k}={metrics['recall']:.4f}, NDCG@{eval_k}={metrics['ndcg']:.4f}")
        else:
            metrics = last_metrics

        recall_history.append(metrics["recall"])
        ndcg_history.append(metrics["ndcg"])

    ckpt_path = os.path.join(cfg["checkpoint_dir"], "lightgcn.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_users": data["num_users"],
            "num_items": data["num_items"],
            "embedding_dim": int(cfg["embedding_dim"]),
            "num_layers": int(cfg["num_layers"]),
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")

    plot_path = os.path.join(cfg["checkpoint_dir"], f"metrics_k{eval_k}.png")
    plot_training_metrics(recall_history, ndcg_history, eval_k, plot_path)
    print(f"Saved metric plot to {plot_path}")


if __name__ == "__main__":
    main()
