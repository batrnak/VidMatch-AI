import argparse
import numpy as np
import torch
import yaml
import os

try:
    from .data import prepare_data, load_mappings
    from .model import LightGCN
    from .utils import get_device
except ImportError:
    from data import prepare_data, load_mappings
    from model import LightGCN
    from utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    if not os.path.exists(config_path):
        fallback_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_path)
        if os.path.exists(fallback_path):
            config_path = fallback_path

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = prepare_data(cfg["data_dir"], int(cfg["min_interactions"]))
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = LightGCN(
        ckpt["num_users"],
        ckpt["num_items"],
        ckpt["embedding_dim"],
        ckpt["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    user_ids, item_ids = load_mappings(cfg["data_dir"])
    if args.user_id not in user_ids:
        raise ValueError("user-id not found in dataset")
    user_index = int(np.where(user_ids == args.user_id)[0][0])

    with torch.no_grad():
        user_emb, item_emb = model.propagate(data["norm_adj"].to(device))
        scores = torch.matmul(user_emb[user_index], item_emb.t()).cpu().numpy()
        seen = data["user_pos"][user_index]
        if len(seen) > 0:
            scores[seen] = -np.inf
        topk_idx = np.argpartition(-scores, kth=args.topk - 1)[: args.topk]
        topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
        topk_item_ids = item_ids[topk_idx]

    print("Top-K movieIds:", topk_item_ids.tolist())


if __name__ == "__main__":
    main()
