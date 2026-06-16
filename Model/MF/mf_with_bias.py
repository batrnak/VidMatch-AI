import sys
from unittest.mock import MagicMock
try:
    import torch
except ImportError:
    sys.modules['torch'] = MagicMock()
    sys.modules['torch.nn'] = MagicMock()
    sys.modules['torch.optim'] = MagicMock()
    sys.modules['torch.sparse'] = MagicMock()

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Adjust path to import LightGCN utilities
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lightgcn-1M", "src"))
try:
    from data import build_eval_data, build_user_pos_list
    from metrics import recall_at_k, ndcg_at_k
except ImportError as e:
    print(f"Error importing LightGCN modules: {e}")
    print("Ensure you run this from the project root or the Model directory.")
    sys.exit(1)


class MF(object):
    """BPR-MF (Bayesian Personalized Ranking Matrix Factorization) optimized for Top-K Ranking"""
    def __init__(self, Y_data, num_users, num_items, eval_data=None, user_pos=None, 
                 K=20, lam=0.01, Winit=None, Xinit=None, learning_rate=0.05, 
                 max_iter=50, print_every=5, eval_k=20):
        
        self.Y_data = Y_data.copy()
        self.num_users = num_users
        self.num_items = num_items
        self.eval_data = eval_data
        self.user_pos = user_pos
        self.K = K
        self.lam = lam
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.print_every = print_every
        self.eval_k = eval_k

        # Initialize Latent Features with small standard deviation
        self.X = np.random.randn(self.num_items, K) * 0.01 if Xinit is None else Xinit
        self.W = np.random.randn(K, self.num_users) * 0.01 if Winit is None else Winit

        # Biases for items (user bias cancels out in BPR)
        self.b = np.zeros(self.num_items)
        self.n_ratings = Y_data.shape[0]

        # History
        self.loss_history = []
        self.recall_history = []
        self.ndcg_history = []

        # Convert user_pos to sets for fast O(1) negative sampling check
        print("Toi uu hoa: Khoi tao tap hop tuong tac de lay mau am nhanh...")
        self.user_pos_set = [set(pos.tolist()) for pos in user_pos]
        print("Khoi tao tap hop hoan tat.")

    def sample_bpr_batch(self, batch_size):
        users = np.random.randint(0, self.num_users, size=batch_size)
        pos_items = np.empty(batch_size, dtype=np.int64)
        neg_items = np.empty(batch_size, dtype=np.int64)
        
        for i, u in enumerate(users):
            pos = self.user_pos[u]
            if len(pos) == 0:
                pos_items[i] = np.random.randint(0, self.num_items)
            else:
                pos_items[i] = pos[np.random.randint(0, len(pos))]
            
            u_set = self.user_pos_set[u]
            while True:
                neg = np.random.randint(0, self.num_items)
                if len(u_set) == 0 or neg not in u_set:
                    neg_items[i] = neg
                    break
        return users, pos_items, neg_items

    def update_params(self, batch_size=8192):
        # 1. Sample triplets (u, i, j)
        users, pos_items, neg_items = self.sample_bpr_batch(batch_size)
        
        # 2. Extract active latent features
        W_u = self.W[:, users].T   # shape (B, K)
        X_i = self.X[pos_items, :]  # shape (B, K)
        X_j = self.X[neg_items, :]  # shape (B, K)
        
        # 3. Predict differences: (W_u^T X_i + b_i) - (W_u^T X_j + b_j)
        scores_pos = np.sum(W_u * X_i, axis=1) + self.b[pos_items]
        scores_neg = np.sum(W_u * X_j, axis=1) + self.b[neg_items]
        diff = scores_pos - scores_neg
        
        # 4. Compute Sigmoid and Gradients
        sigmoid_val = 1.0 / (1.0 + np.exp(-np.clip(diff, -50.0, 50.0)))
        c = sigmoid_val - 1.0  # shape (B,)
        
        grad_W = c[:, None] * (X_i - X_j) + self.lam * W_u
        grad_Xi = c[:, None] * W_u + self.lam * X_i
        grad_Xj = -c[:, None] * W_u + self.lam * X_j
        
        grad_bi = c + self.lam * self.b[pos_items]
        grad_bj = -c + self.lam * self.b[neg_items]
        
        # 5. Apply SGD updates using np.add.at for duplicate indices in batch
        np.add.at(self.W, (slice(None), users), -self.learning_rate * grad_W.T)
        np.add.at(self.X, pos_items, -self.learning_rate * grad_Xi)
        np.add.at(self.X, neg_items, -self.learning_rate * grad_Xj)
        
        np.add.at(self.b, pos_items, -self.learning_rate * grad_bi)
        np.add.at(self.b, neg_items, -self.learning_rate * grad_bj)
        
        loss = -np.mean(np.log(sigmoid_val + 1e-8))
        return loss

    def evaluate_ranking(self):
        """Dự đoán Ranking sử dụng BPR-MF score: W_u^T X_i + b_i"""
        full_scores = self.W.T @ self.X.T + self.b.reshape(1, -1)
        
        recalls = []
        ndcgs = []
        users = np.array(list(self.eval_data.keys()), dtype=np.int64)
        
        for u in users:
            scores_u = full_scores[u, :].copy()
            seen = self.user_pos[u]
            if len(seen) > 0:
                scores_u[seen] = -np.inf
                
            topk_idx = np.argpartition(-scores_u, kth=self.eval_k - 1)[:self.eval_k]
            topk_scores = scores_u[topk_idx]
            topk_order = np.argsort(-topk_scores)
            ranklist = topk_idx[topk_order]
            
            gt = self.eval_data[u]
            recalls.append(recall_at_k(ranklist, gt, self.eval_k))
            ndcgs.append(ndcg_at_k(ranklist, gt, self.eval_k))
            
        return float(np.mean(recalls)), float(np.mean(ndcgs))

    def fit(self):
        print(f"Bắt đầu huấn luyện BPR-MF và đánh giá Top-{self.eval_k} Ranking...")
        batch_size = 8192
        steps = self.n_ratings // batch_size
        
        for it in range(self.max_iter):
            epoch_loss = 0
            for _ in range(steps):
                loss_val = self.update_params(batch_size)
                epoch_loss += loss_val
            epoch_loss /= steps
            self.loss_history.append(epoch_loss)
            
            if (it + 1) % self.print_every == 0 or it == 0 or it == self.max_iter - 1:
                if self.eval_data is not None:
                    recall, ndcg = self.evaluate_ranking()
                    self.recall_history.append((it + 1, recall))
                    self.ndcg_history.append((it + 1, ndcg))
                    print(f'Iter {it+1}/{self.max_iter} - Loss: {epoch_loss:.4f} - Recall@{self.eval_k}: {recall:.4f} - NDCG@{self.eval_k}: {ndcg:.4f}')
                else:
                    print(f'Iter {it+1}/{self.max_iter} - Loss: {epoch_loss:.4f}')


def main():
    parser = argparse.ArgumentParser(description='Train BPR-MF for Top-K Ranking')
    parser.add_argument('--k', type=int, default=20, help='Latent factors (K)')
    parser.add_argument('--lam', type=float, default=0.01, help='Regularization lambda')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--max-iter', type=int, default=30, help='Max iterations')
    parser.add_argument('--print-every', type=int, default=5, help='Print every N iterations')
    parser.add_argument('--eval-k', type=int, default=20, help='Top-K for Ranking metrics')
    parser.add_argument('--plot-path', type=str, default='checkpoints/mf_metrics_k20.png', help='Path to save metrics plot')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints/mf.pt', help='Path to save MF checkpoint')
    args = parser.parse_args()

    # Locate Datasets dir
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(proj_root, "datasets", "ml-1m")

    print('=== THÔNG TIN CẤU HÌNH BPR-MF RANKING ===')
    print(f'Latent Factors (K) = {args.k}')
    print(f'Lambda = {args.lam}')
    print(f'Learning Rate = {args.lr}')
    print(f'Max Iterations = {args.max_iter}')
    print(f'Top-K Ranking = {args.eval_k}')
    print('=========================================\n')

    # Prepare Data with Leave-One-Out split
    print("Loading pre-processed CSV data...")
    import json
    processed_dir = os.path.join(data_dir, "processed")
    with open(os.path.join(processed_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
        
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    
    Y_data = train_df[["userId", "movieId", "rating"]].values.astype(np.float64)
    eval_data = build_eval_data(metadata["num_users"], test_df)
    user_pos = build_user_pos_list(metadata["num_users"], train_df)

    print(f'MovieLens 1M - Train ratings: {Y_data.shape[0]} | Test users: {len(eval_data)}')
    
    # Initialize and Train Model
    rs = MF(Y_data, num_users=metadata["num_users"], num_items=metadata["num_items"], 
            eval_data=eval_data, user_pos=user_pos, 
            K=args.k, lam=args.lam, print_every=args.print_every,
            learning_rate=args.lr, max_iter=args.max_iter, eval_k=args.eval_k)
            
    rs.fit()

    # Save Checkpoint Plot
    plot_file = Path(os.path.join(proj_root, args.plot_path))
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    if len(rs.recall_history) > 0:
        iters, recalls = zip(*rs.recall_history)
        _, ndcgs = zip(*rs.ndcg_history)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(iters, recalls, 'b-o', label=f'Recall@{args.eval_k}')
        plt.xlabel('Iteration')
        plt.ylabel(f'Recall@{args.eval_k}')
        plt.title('BPR-MF Ranking Performance (Recall)')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(iters, ndcgs, 'g-s', label=f'NDCG@{args.eval_k}')
        plt.xlabel('Iteration')
        plt.ylabel(f'NDCG@{args.eval_k}')
        plt.title('BPR-MF Ranking Performance (NDCG)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(plot_file)
        print(f'\nRanking metrics plot saved to: {plot_file.resolve()}')

    # Save checkpoint weights (using torch if available, otherwise numpy npz)
    ckpt_path = os.path.join(proj_root, args.checkpoint_path)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    
    try:
        import torch
        if 'unittest.mock' in str(type(torch)):
            raise ImportError("PyTorch is mocked")
        torch.save({
            'user_embeddings': torch.from_numpy(rs.W.T).float(),
            'item_embeddings': torch.from_numpy(rs.X).float(),
            'user_biases': torch.zeros(rs.num_users).float(),
            'item_biases': torch.from_numpy(rs.b).float(),
            'global_mean': 0.0,
            'user_means': torch.zeros(rs.num_items).float()
        }, ckpt_path)
        print(f'Saved BPR-MF PyTorch checkpoint to: {ckpt_path}')
    except Exception as e:
        npz_path = ckpt_path.replace('.pt', '.npz')
        np.savez(npz_path, 
                 user_embeddings=rs.W.T,
                 item_embeddings=rs.X,
                 user_biases=np.zeros(rs.num_users),
                 item_biases=rs.b,
                 global_mean=0.0,
                 user_means=np.zeros(rs.num_items))
        print(f'Could not save PyTorch checkpoint ({e}). Saved numpy checkpoint to: {npz_path}')

if __name__ == '__main__':
    main()
