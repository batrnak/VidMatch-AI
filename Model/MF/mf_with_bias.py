import os
import zipfile
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

class MF(object):
    """Matrix Factorization with User and Item Biases"""
    def __init__(self, Y_data, K, lam=0.1, Xinit=None, Winit=None,
                 learning_rate=0.5, max_iter=100, print_every=10, user_based=0):
        self.Y_raw = Y_data.copy()
        self.Y_data = Y_data.copy()
        self.K = K
        self.lam = lam
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.print_every = print_every
        self.user_based = user_based
        
        # Số lượng users và items
        self.n_users = int(np.max(Y_data[:, 0])) + 1
        self.n_items = int(np.max(Y_data[:, 1])) + 1

        if Xinit is None:
            self.X = np.random.randn(self.n_items, K)
        else:
            self.X = Xinit

        if Winit is None:
            self.W = np.random.randn(K, self.n_users)
        else:
            self.W = Winit

        # Biases
        self.b = np.random.randn(self.n_items)
        self.d = np.random.randn(self.n_users)
        self.n_ratings = Y_data.shape[0]
        self.mu = 0
        
        # Lịch sử để vẽ biểu đồ
        self.loss_history = []
        self.rmse_train_history = []
        self.rmse_test_history = []

        # Tối ưu hóa: Nhóm sẵn các chỉ mục dòng (row indices) cho mỗi user và item để tăng tốc truy vấn gấp 100 lần
        print("Toi uu hoa: Dang lap chi muc tuong tac User/Item...")
        self.user_to_rating_indices = {u: [] for u in range(self.n_users)}
        self.item_to_rating_indices = {i: [] for i in range(self.n_items)}
        for idx in range(self.n_ratings):
            u = int(Y_data[idx, 0])
            i = int(Y_data[idx, 1])
            self.user_to_rating_indices[u].append(idx)
            self.item_to_rating_indices[i].append(idx)
            
        # Chuyển đổi sang numpy array để slice cực nhanh
        self.user_to_rating_indices = {u: np.array(indices, dtype=np.int32) for u, indices in self.user_to_rating_indices.items()}
        self.item_to_rating_indices = {i: np.array(indices, dtype=np.int32) for i, indices in self.item_to_rating_indices.items()}
        print("Lap chi muc hoan tat.")

    def normalize_Y(self):
        if self.user_based:
            user_col = 0
            n_objects = self.n_users
        else:
            user_col = 1
            n_objects = self.n_items

        users = self.Y_data[:, user_col]
        self.muu = np.zeros((n_objects,))
        for n in range(n_objects):
            ids = np.where(users == n)[0]
            if len(ids) == 0:
                continue
            ratings = self.Y_data[ids, 2]
            m = np.mean(ratings)
            if np.isnan(m):
                m = 0
            self.muu[n] = m
            self.Y_data[ids, 2] = ratings - m

    def loss(self):
        # Vectorized loss computation for high performance
        users = self.Y_data[:, 0].astype(np.int32)
        items = self.Y_data[:, 1].astype(np.int32)
        ratings = self.Y_data[:, 2]
        
        preds = np.sum(self.X[items, :] * self.W[:, users].T, axis=1) + self.b[items] + self.d[users] + self.mu
        L = 0.5 * np.sum((preds - ratings) ** 2) / self.n_ratings
        
        # Regularization term
        L += 0.5 * self.lam * (
            np.linalg.norm(self.X, 'fro') + 
            np.linalg.norm(self.W, 'fro') + 
            np.linalg.norm(self.b) + 
            np.linalg.norm(self.d)
        )
        return L

    def get_items_rated_by_user(self, user_id):
        ids = self.user_to_rating_indices[user_id]
        item_ids = self.Y_data[ids, 1].astype(np.int32)
        ratings = self.Y_data[ids, 2]
        return item_ids, ratings

    def get_users_who_rate_item(self, item_id):
        ids = self.item_to_rating_indices[item_id]
        user_ids = self.Y_data[ids, 0].astype(np.int32)
        ratings = self.Y_data[ids, 2]
        return user_ids, ratings

    def updateX(self):
        for m in range(self.n_items):
            user_ids, ratings = self.get_users_who_rate_item(m)
            if len(user_ids) == 0:
                continue

            Wm = self.W[:, user_ids]
            dm = self.d[user_ids]
            xm = self.X[m, :]

            error = xm.dot(Wm) + self.b[m] + dm + self.mu - ratings

            grad_xm = error.dot(Wm.T)/self.n_ratings + self.lam*xm
            grad_bm = np.sum(error)/self.n_ratings + self.lam*self.b[m]
            self.X[m, :] -= self.learning_rate*grad_xm.reshape((self.K,))
            self.b[m]    -= self.learning_rate*grad_bm

    def updateW(self):
        for n in range(self.n_users):
            item_ids, ratings = self.get_items_rated_by_user(n)
            if len(item_ids) == 0:
                continue
            Xn = self.X[item_ids, :]
            bn = self.b[item_ids]
            wn = self.W[:, n]

            error = Xn.dot(wn) + bn + self.mu + self.d[n] - ratings
            grad_wn = Xn.T.dot(error)/self.n_ratings + self.lam*wn
            grad_dn = np.sum(error)/self.n_ratings + self.lam*self.d[n]
            self.W[:, n] -= self.learning_rate*grad_wn.reshape((self.K,))
            self.d[n]    -= self.learning_rate*grad_dn

    def fit(self, Y_test=None):
        self.normalize_Y()
        for it in range(self.max_iter):
            self.updateX()
            self.updateW()
            
            # Lưu lại lịch sử hội tụ
            if (it + 1) % self.print_every == 0 or it == 0 or it == self.max_iter - 1:
                loss_val = self.loss()
                rmse_train = self.evaluate_RMSE(self.Y_raw)
                self.loss_history.append((it + 1, loss_val))
                self.rmse_train_history.append((it + 1, rmse_train))
                
                if Y_test is not None:
                    rmse_test = self.evaluate_RMSE(Y_test)
                    self.rmse_test_history.append((it + 1, rmse_test))
                    
                if (it + 1) % self.print_every == 0 or it == 0 or it == self.max_iter - 1:
                    if Y_test is not None:
                        print(f'Iter {it+1}/{self.max_iter} - Loss: {loss_val:.6f} - RMSE train: {rmse_train:.6f} - RMSE test: {rmse_test:.6f}')
                    else:
                        print(f'Iter {it+1}/{self.max_iter} - Loss: {loss_val:.6f} - RMSE train: {rmse_train:.6f}')

    def pred(self, u, i):
        u = int(u)
        i = int(i)
        if u >= self.n_users or i >= self.n_items:
            return 3.0
        if self.user_based == 1:
            bias = self.muu[u]
        else:
            bias = self.muu[i]

        pred = self.X[i, :].dot(self.W[:, u]) + self.b[i] + self.d[u] + bias
        return float(np.clip(pred, 0, 5))

    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        users = rate_test[:, 0].astype(np.int32)
        items = rate_test[:, 1].astype(np.int32)
        ratings = rate_test[:, 2]
        
        # Vectorized evaluation for fast calculation
        u_clipped = np.clip(users, 0, self.n_users - 1)
        i_clipped = np.clip(items, 0, self.n_items - 1)
        
        preds = np.sum(self.X[i_clipped, :] * self.W[:, u_clipped].T, axis=1) + self.b[i_clipped] + self.d[u_clipped]
        if self.user_based == 1:
            preds += self.muu[u_clipped]
        else:
            preds += self.muu[i_clipped]
            
        preds = np.clip(preds, 0, 5)
        SE = np.sum((preds - ratings) ** 2)
        return np.sqrt(SE / n_tests)

def resolve_project_path(*parts):
    rel_path = Path(*parts)
    if rel_path.is_absolute() and rel_path.exists():
        return rel_path

    candidates = [Path.cwd()] + list(Path.cwd().parents)
    checked = []
    for root in candidates:
        if (root / 'datasets').exists():
            path = root.joinpath(*parts)
            checked.append(path)
            if path.exists():
                return path

    fallback = Path.cwd().joinpath(*parts)
    checked.append(fallback)
    if fallback.exists():
        return fallback

    checked_msg = '\n'.join(str(p) for p in checked)
    raise FileNotFoundError(
        f'Could not find {rel_path}. Current working dir: {Path.cwd()}\nChecked:\n{checked_msg}'
    )

def download_and_extract_dataset(url: str, folder_name: str):
    """Tự động tải và giải nén tập dữ liệu nếu chưa tồn tại"""
    try:
        datasets_dir = resolve_project_path('datasets')
    except FileNotFoundError:
        datasets_dir = Path.cwd().parent / 'datasets' if (Path.cwd().parent / 'docker-compose.yml').exists() else Path.cwd() / 'datasets'
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
    target_dir = datasets_dir / folder_name
    if target_dir.exists():
        return target_dir
        
    zip_path = datasets_dir / f'{folder_name}.zip'
    print(f'==> Khong tim thay {folder_name}. Dang tai tu GroupLens...')
    print(f'URL: {url}')
    urllib.request.urlretrieve(url, zip_path)
    
    print(f'==> Dang giai nen {zip_path.name}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(datasets_dir)
    zip_path.unlink() # Xóa file zip sau khi giải nén
    print(f'==> Da tai va giai nen thanh cong tai: {target_dir}')
    return target_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Matrix Factorization with Bias on MovieLens 1M')
    parser.add_argument('--k', type=int, default=20, help='Latent factors (K)')
    parser.add_argument('--lam', type=float, default=0.05, help='Regularization lambda')
    parser.add_argument('--lr', type=float, default=0.5, help='Learning rate')
    parser.add_argument('--max-iter', type=int, default=30, help='Max iterations')
    parser.add_argument('--print-every', type=int, default=5, help='Print every N iterations')
    parser.add_argument('--user-based', type=int, default=0, help='User-based normalization (1) or Item-based (0)')
    parser.add_argument('--plot-path', type=str, default='checkpoints/mf_convergence_1m.png', help='Path to save convergence plot')
    args = parser.parse_args()

    print('=== THONG TIN CAU HINH ===')
    print(f'K = {args.k}')
    print(f'Lambda = {args.lam}')
    print(f'Learning Rate = {args.lr}')
    print(f'Max Iter = {args.max_iter}')
    print(f'User Based = {args.user_based}')
    print('==========================\n')

    # Tự động tải MovieLens 1M nếu thiếu
    download_and_extract_dataset('https://files.grouplens.org/datasets/movielens/ml-1m.zip', 'ml-1m')

    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings_path = resolve_project_path('datasets', 'ml-1m', 'ratings.dat')
    print(f'Loading ratings from: {ratings_path}')
    ratings_1m = pd.read_csv(ratings_path, sep='::', names=r_cols, encoding='latin-1', engine='python')

    rate_all = ratings_1m[['user_id', 'movie_id', 'rating']].values.copy()
    
    # Chuyển chỉ mục về 0-indexed
    rate_all[:, :2] -= 1

    # Chia Train/Test (80% train, 20% test)
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(rate_all))
    split_idx = int(0.8 * len(rate_all))
    rate_train = rate_all[shuffled_indices[:split_idx]]
    rate_test = rate_all[shuffled_indices[split_idx:]]

    print(f'MovieLens 1M - Train size: {rate_train.shape[0]} | Test size: {rate_test.shape[0]}')
    
    print('Starting training...')
    rs = MF(rate_train, K=args.k, lam=args.lam, print_every=args.print_every,
            learning_rate=args.lr, max_iter=args.max_iter, user_based=args.user_based)
    rs.fit(Y_test=rate_test)

    final_rmse = rs.evaluate_RMSE(rate_test)
    print(f'\nTraining completed! Final Test RMSE = {final_rmse:.6f}')

    # Đảm bảo thư mục lưu checkpoint tồn tại
    plot_file = Path(args.plot_path)
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    # Vẽ và lưu biểu đồ hội tụ
    iters, losses = zip(*rs.loss_history)
    _, train_rmses = zip(*rs.rmse_train_history)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(iters, losses, 'r-o', label='Train Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Convergence (MovieLens 1M)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(iters, train_rmses, 'b-s', label='Train RMSE')
    if len(rs.rmse_test_history) > 0:
        _, test_rmses = zip(*rs.rmse_test_history)
        plt.plot(iters, test_rmses, 'g-d', label='Test RMSE')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('RMSE Convergence (MovieLens 1M)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_file)
    print(f'Convergence plot saved to: {plot_file.resolve()}')
