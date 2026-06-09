# LightGCN on MovieLens 1M

Minimal LightGCN implementation for MovieLens 1M (ratings.dat).

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

## Train

```
python -m src.train --config config.yaml
```

This will download MovieLens 1M into `data/ml-1m/` (if missing), build the graph, train, and save a checkpoint in `checkpoints/`.

After training, a metric chart is also saved to `checkpoints/metrics_k<eval_k>.png` showing Recall@K and NDCG@K by epoch on the test split.

## Evaluate

```
python -m src.eval --config config.yaml --checkpoint checkpoints/lightgcn.pt
```

## Inference (top-N recommendations)

```
python -m src.infer --config config.yaml --checkpoint checkpoints/lightgcn.pt --user-id 123 --topk 10
```

`--user-id` is the raw `userId` from MovieLens.

## Notes

- MovieLens 1M is much smaller; training on CPU is feasible but a GPU will still speed up epochs. 4GB+ RAM is usually sufficient.
- The split uses leave-one-out (last interaction as test). Users with too few interactions may have no validation item.
