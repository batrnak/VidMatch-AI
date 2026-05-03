import numpy as np


def recall_at_k(ranklist: np.ndarray, ground_truth: int, k: int) -> float:
    return float(ground_truth in ranklist[:k])


def ndcg_at_k(ranklist: np.ndarray, ground_truth: int, k: int) -> float:
    if ground_truth not in ranklist[:k]:
        return 0.0
    index = np.where(ranklist[:k] == ground_truth)[0][0]
    return float(1.0 / np.log2(index + 2))
