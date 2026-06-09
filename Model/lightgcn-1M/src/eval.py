import argparse
import yaml
import torch
import os

try:
    from .data import prepare_data, build_eval_data
    from .model import LightGCN
    from .train import evaluate
    from .utils import get_device
except ImportError:
    from data import prepare_data, build_eval_data
    from model import LightGCN
    from train import evaluate
    from utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
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

    norm_adj = data["norm_adj"].to(device)
    eval_data = build_eval_data(data["num_users"], data["test_df"])

    metrics = evaluate(
        model,
        norm_adj,
        eval_data,
        data["user_pos"],
        data["num_items"],
        int(cfg["eval_k"]),
        int(cfg["eval_batch_size"]),
        device,
    )
    print(f"Recall@{cfg['eval_k']}={metrics['recall']:.4f}, NDCG@{cfg['eval_k']}={metrics['ndcg']:.4f}")


if __name__ == "__main__":
    main()
