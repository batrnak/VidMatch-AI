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


def calculate_val_loss(
    model: LightGCN,
    norm_adj: torch.sparse.FloatTensor,
    eval_data: Dict[int, int],
    user_pos_set: list[set],
    num_items: int,
    batch_size: int,
    reg_weight: float,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        val_users = list(eval_data.keys())
        if len(val_users) == 0:
            return 0.0
        
        # Sample validation users
        sampled_users = np.random.choice(val_users, size=min(batch_size, len(val_users)), replace=False)
        pos_items = np.empty(len(sampled_users), dtype=np.int64)
        neg_items = np.empty(len(sampled_users), dtype=np.int64)
        
        for i, u in enumerate(sampled_users):
            pos_items[i] = eval_data[u]
            u_set = user_pos_set[u]
            val_item = eval_data[u]
            while True:
                neg = np.random.randint(0, num_items)
                if neg not in u_set and neg != val_item:
                    neg_items[i] = neg
                    break
        
        users_t = torch.from_numpy(sampled_users).to(device)
        pos_t = torch.from_numpy(pos_items).to(device)
        neg_t = torch.from_numpy(neg_items).to(device)
        
        user_emb, item_emb = model.propagate(norm_adj)
        loss = model.bpr_loss(
            users_t,
            pos_t,
            neg_t,
            user_emb,
            item_emb,
            reg_weight,
        )
        return float(loss.item())


def plot_loss_curve(train_losses: list[float], val_losses: list[float], output_path: str) -> None:
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot(epochs, train_losses, 'r-', linewidth=2, label='Train BPR Loss')
    ax.plot(epochs, val_losses, 'b--', linewidth=2, label='Val BPR Loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BPR Loss")
    ax.set_title("LightGCN Training & Validation Loss Curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_metrics(
    epochs: list[int], recalls: list[float], ndcgs: list[float], k: int, output_path: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)

    axes[0].plot(epochs, recalls, marker="o", linewidth=1.8)
    axes[0].set_title(f"Recall@{k} on Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(f"Recall@{k}")
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(epochs)

    axes[1].plot(epochs, ndcgs, marker="o", linewidth=1.8, color="tab:orange")
    axes[1].set_title(f"NDCG@{k} on Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(f"NDCG@{k}")
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(epochs)

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

    eval_data = build_eval_data(data["num_users"], data["val_df"])

    steps_per_epoch = max(1, len(data["train_df"]) // int(cfg["batch_size"]))
    recall_history = []
    ndcg_history = []
    eval_epochs = []
    eval_k = int(cfg["eval_k"])

    train_loss_history = []
    val_loss_history = []

    print(f"Starting training on {device}...")
    last_metrics = {"recall": 0.0, "ndcg": 0.0}
    eval_every = 10

    # Early Stopping configuration
    best_recall = 0.0
    best_epoch = 0
    patience = 10  # Số lần đánh giá liên tiếp không cải thiện trước khi dừng
    patience_counter = 0
    best_state_dict = None

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch}")
        epoch_loss = 0.0
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
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": float(loss.item())})

        epoch_loss /= steps_per_epoch
        train_loss_history.append(epoch_loss)

        # Tính Val BPR Loss mỗi epoch
        val_loss = calculate_val_loss(
            model,
            norm_adj,
            eval_data,
            data["user_pos_set"],
            data["num_items"],
            int(cfg["batch_size"]),
            float(cfg["reg_weight"]),
            device
        )
        val_loss_history.append(val_loss)

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
            print(f"Epoch {epoch}: Val Recall@{eval_k}={metrics['recall']:.4f}, Val NDCG@{eval_k}={metrics['ndcg']:.4f} | Train Loss={epoch_loss:.4f}, Val Loss={val_loss:.4f}")
            eval_epochs.append(epoch)
            recall_history.append(metrics["recall"])
            ndcg_history.append(metrics["ndcg"])

            # Early Stopping: lưu model tốt nhất và kiểm tra patience
            if metrics["recall"] > best_recall:
                best_recall = metrics["recall"]
                best_epoch = epoch
                patience_counter = 0
                # Lưu bản sao trọng số tốt nhất
                import copy
                best_state_dict = copy.deepcopy(model.state_dict())
                print(f"  -> New best Recall@{eval_k}={best_recall:.4f} at epoch {epoch}")
            else:
                patience_counter += 1
                print(f"  -> No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}! Best Recall@{eval_k}={best_recall:.4f} at epoch {best_epoch}")
                break

    # Lưu checkpoint tốt nhất (best model) vào best_lightgcn.pt
    if best_state_dict is not None:
        best_ckpt_path = os.path.join(cfg["checkpoint_dir"], "best_lightgcn.pt")
        torch.save(
            {
                "model_state": best_state_dict,
                "num_users": data["num_users"],
                "num_items": data["num_items"],
                "embedding_dim": int(cfg["embedding_dim"]),
                "num_layers": int(cfg["num_layers"]),
                "best_recall": best_recall,
                "best_epoch": best_epoch,
            },
            best_ckpt_path,
        )
        print(f"Saved BEST checkpoint (epoch {best_epoch}) to {best_ckpt_path}")

    # Lưu checkpoint cuối cùng (last model) vào lightgcn.pt
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
    print(f"Saved LAST checkpoint to {ckpt_path}")

    # Đánh giá cuối cùng trên tập Test
    print("\n=== ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST (KHÔNG RÒ RỈ DỮ LIỆU) ===")
    test_eval_data = build_eval_data(data["num_users"], data["test_df"])
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best model weights from epoch {best_epoch} for test evaluation.")
    test_metrics = evaluate(
        model,
        norm_adj,
        test_eval_data,
        user_pos,
        data["num_items"],
        eval_k,
        int(cfg["eval_batch_size"]),
        device,
    )
    print(f"LightGCN Final Test Results: Recall@{eval_k}={test_metrics['recall']:.4f}, NDCG@{eval_k}={test_metrics['ndcg']:.4f}")
    print("============================================================\n")

    plot_path = os.path.join(cfg["checkpoint_dir"], f"metrics_k{eval_k}.png")
    plot_training_metrics(eval_epochs, recall_history, ndcg_history, eval_k, plot_path)
    print(f"Saved metric plot to {plot_path}")

    loss_plot_path = os.path.join(cfg["checkpoint_dir"], "lightgcn_loss.png")
    plot_loss_curve(train_loss_history, val_loss_history, loss_plot_path)
    print(f"Saved loss curve plot to {loss_plot_path}")


if __name__ == "__main__":
    main()
