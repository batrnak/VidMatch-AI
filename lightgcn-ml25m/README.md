# LightGCN on MovieLens 25M

Minimal LightGCN implementation for MovieLens 25M (ratings.csv).

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

This will download MovieLens 25M into `data/ml-25m/` (if missing), build the graph, train, and save a checkpoint in `checkpoints/`.

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

- MovieLens 25M is large. CPU training is slow and memory heavy. If possible, use a GPU and enough RAM (>= 16GB recommended).
- The split uses leave-one-out (last interaction as test). Users with too few interactions may have no validation item.
