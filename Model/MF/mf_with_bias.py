import os
import sys
import argparse
from pathlib import Path
import zipfile
import urllib.request
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
    """Matrix Factorization with User and Item Biases optimized for Top-K Ranking"""
    def __init__(self, Y_data, num_users, num_items, eval_data=None, user_pos=None, 
                 K=20, lam=0.1, Winit=None, Xinit=None, learning_rate=0.5, 
                 max_iter=100, print_every=10, user_based=0, eval_k=20):
        
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
        self.user_based = user_based
        self.eval_k = eval_k

        # Initialize Latent Features
        self.X = np.random.randn(self.num_items, K) if Xinit is None else Xinit
        self.W = np.random.randn(K, self.num_users) if Winit is None else Winit

        # Biases
        self.b = np.random.randn(self.num_items)
        self.d = np.random.randn(self.num_users)
        self.mu = 0
        self.muu = np.zeros(self.num_items if user_based == 0 else self.num_users)
        self.n_ratings = Y_data.shape[0]

        # History
        self.loss_history = []
        self.recall_history = []
        self.ndcg_history = []

        # Fast Interaction Indexing for SGD
        print("Toi uu hoa: Dang lap chi muc tuong tac User/Item...")
        self.user_to_rating_indices = {u: [] for u in range(self.num_users)}
        self.item_to_rating_indices = {i: [] for i in range(self.num_items)}
        for idx in range(self.n_ratings):
            u = int(Y_data[idx, 0])
            i = int(Y_data[idx, 1])
            self.user_to_rating_indices[u].append(idx)
            self.item_to_rating_indices[i].append(idx)
            
        self.user_to_rating_indices = {u: np.array(indices, dtype=np.int32) for u, indices in self.user_to_rating_indices.items()}
        self.item_to_rating_indices = {i: np.array(indices, dtype=np.int32) for i, indices in self.item_to_rating_indices.items()}
        print("Lap chi muc hoan tat.")

    def normalize_Y(self):
        user_col = 0 if self.user_based else 1
        n_objects = self.num_users if self.user_based else self.num_items
        users = self.Y_data[:, user_col]
        for n in range(n_objects):
            ids = np.where(users == n)[0]
            if len(ids) == 0:
                continue
            ratings = self.Y_data[ids, 2]
            m = np.mean(ratings)
            if np.isnan(m): m = 0
            self.muu[n] = m
            self.Y_data[ids, 2] = ratings - m

    def loss(self):
        users = self.Y_data[:, 0].astype(np.int32)
        items = self.Y_data[:, 1].astype(np.int32)
        ratings = self.Y_data[:, 2]
        
        preds = np.sum(self.X[items, :] * self.W[:, users].T, axis=1) + self.b[items] + self.d[users] + self.mu
        L = 0.5 * np.sum((preds - ratings) ** 2) / self.n_ratings
        
        # Regularization
        L += 0.5 * self.lam * (np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro') + 
                               np.linalg.norm(self.b) + np.linalg.norm(self.d))
        return L

    def update_params(self):
        # Update X
        for m in range(self.num_items):
            ids = self.item_to_rating_indices[m]
            if len(ids) == 0: continue
            u_ids = self.Y_data[ids, 0].astype(np.int32)
            ratings = self.Y_data[ids, 2]
            Wm = self.W[:, u_ids]
            dm = self.d[u_ids]
            xm = self.X[m, :]
            error = xm.dot(Wm) + self.b[m] + dm + self.mu - ratings
            grad_xm = error.dot(Wm.T)/self.n_ratings + self.lam*xm
            grad_bm = np.sum(error)/self.n_ratings + self.lam*self.b[m]
            self.X[m, :] -= self.learning_rate*grad_xm.reshape((self.K,))
            self.b[m] -= self.learning_rate*grad_bm

        # Update W
        for n in range(self.num_users):
            ids = self.user_to_rating_indices[n]
            if len(ids) == 0: continue
            i_ids = self.Y_data[ids, 1].astype(np.int32)
            ratings = self.Y_data[ids, 2]
            Xn = self.X[i_ids, :]
            bn = self.b[i_ids]
            wn = self.W[:, n]
            error = Xn.dot(wn) + bn + self.mu + self.d[n] - ratings
            grad_wn = Xn.T.dot(error)/self.n_ratings + self.lam*wn
            grad_dn = np.sum(error)/self.n_ratings + self.lam*self.d[n]
            self.W[:, n] -= self.learning_rate*grad_wn.reshape((self.K,))
            self.d[n] -= self.learning_rate*grad_dn

    def evaluate_ranking(self):
        """Dự đoán Ranking thay vì đoán điểm số trực tiếp (RMSE)"""
        full_scores = self.W.T @ self.X.T
        full_scores += self.b.reshape(1, -1)
        full_scores += self.d.reshape(-1, 1)
        
        if self.user_based == 1:
            full_scores += self.muu.reshape(-1, 1)
        else:
            full_scores += self.muu.reshape(1, -1)
            
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
        self.normalize_Y()
        
        print(f"Bắt đầu huấn luyện MF và đánh giá Top-{self.eval_k} Ranking...")
        for it in range(self.max_iter):
            self.update_params()
            
            if (it + 1) % self.print_every == 0 or it == 0 or it == self.max_iter - 1:
                loss_val = self.loss()
                self.loss_history.append(loss_val)
                
                if self.eval_data is not None:
                    recall, ndcg = self.evaluate_ranking()
                    self.recall_history.append((it + 1, recall))
                    self.ndcg_history.append((it + 1, ndcg))
                    print(f'Iter {it+1}/{self.max_iter} - Loss: {loss_val:.4f} - Recall@{self.eval_k}: {recall:.4f} - NDCG@{self.eval_k}: {ndcg:.4f}')
                else:
                    print(f'Iter {it+1}/{self.max_iter} - Loss: {loss_val:.4f}')


def main():
    parser = argparse.ArgumentParser(description='Train Matrix Factorization for Top-K Ranking')
    parser.add_argument('--k', type=int, default=20, help='Latent factors (K)')
    parser.add_argument('--lam', type=float, default=0.05, help='Regularization lambda')
    parser.add_argument('--lr', type=float, default=0.5, help='Learning rate')
    parser.add_argument('--max-iter', type=int, default=30, help='Max iterations')
    parser.add_argument('--print-every', type=int, default=5, help='Print every N iterations')
    parser.add_argument('--eval-k', type=int, default=20, help='Top-K for Ranking metrics')
    parser.add_argument('--plot-path', type=str, default='checkpoints/mf_metrics_k20.png', help='Path to save metrics plot')
    args = parser.parse_args()

    # Locate Datasets dir
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(proj_root, "datasets", "ml-1m")

    print('=== THÔNG TIN CẤU HÌNH MF RANKING ===')
    print(f'Latent Factors (K) = {args.k}')
    print(f'Lambda = {args.lam}')
    print(f'Learning Rate = {args.lr}')
    print(f'Max Iterations = {args.max_iter}')
    print(f'Top-K Ranking = {args.eval_k}')
    print('=====================================\n')

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
        plt.title('MF Ranking Performance (Recall)')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(iters, ndcgs, 'g-s', label=f'NDCG@{args.eval_k}')
        plt.xlabel('Iteration')
        plt.ylabel(f'NDCG@{args.eval_k}')
        plt.title('MF Ranking Performance (NDCG)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(plot_file)
        print(f'\nRanking metrics plot saved to: {plot_file.resolve()}')

if __name__ == '__main__':
    main()
