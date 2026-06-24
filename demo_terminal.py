import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import yaml

# ANSI Escape Codes for coloring
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
WHITE = "\033[97m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_header(title):
    width = 110
    print(f"\n{CYAN}{'=' * width}{RESET}")
    print(f"{CYAN}{BOLD}{title.center(width)}{RESET}")
    print(f"{CYAN}{'=' * width}{RESET}\n")

def main():
    proj_root = os.path.dirname(os.path.abspath(__file__))
    
    print_header("VidMatch-AI: KHỞI ĐỘNG HỆ THỐNG DEMO GỢI Ý PHIM")
    
    # 1. Load mappings
    print("1/5 Đang tải ánh xạ User/Item mappings...")
    mappings_path = os.path.join(proj_root, "data", "mappings.npz")
    if not os.path.exists(mappings_path):
        print(f"{RED}Lỗi: Không tìm thấy file {mappings_path}. Hãy chạy tiền xử lý trước.{RESET}")
        return
    mappings = np.load(mappings_path)
    user_ids = mappings["user_ids"]
    item_ids = mappings["item_ids"]
    num_users = len(user_ids)
    num_items = len(item_ids)
    
    # 2. Load movie information
    print("2/5 Đang tải thông tin phim từ movies.dat...")
    movies_path = os.path.join(proj_root, "datasets", "ml-1m", "movies.dat")
    movie_info = {}
    if os.path.exists(movies_path):
        with open(movies_path, "r", encoding="latin1") as f:
            for line in f:
                parts = line.strip().split("::")
                if len(parts) >= 3:
                    m_id = int(parts[0])
                    title = parts[1]
                    genres = parts[2]
                    movie_info[m_id] = (title, genres)
    else:
        print(f"{YELLOW}Cảnh báo: Không tìm thấy file {movies_path}. Tên phim sẽ bị ẩn.{RESET}")

    # 3. Load datasets
    print("3/5 Đang đọc tập dữ liệu Train/Val/Test...")
    processed_dir = os.path.join(proj_root, "datasets", "ml-1m", "processed")
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    
    # Build seen set for each user
    user_seen = [set() for _ in range(num_users)]
    for row in train_df.itertuples(index=False):
        user_seen[row.userId].add(row.movieId)
        
    # Build validation and test dicts
    val_dict = {row.userId: row.movieId for row in val_df.itertuples(index=False)}
    test_dict = {row.userId: row.movieId for row in test_df.itertuples(index=False)}

    # 4. Load BPR-MF model
    print("4/5 Đang tải trọng số mô hình BPR-MF...")
    mf_path = os.path.join(proj_root, "checkpoints", "mf.npz")
    if not os.path.exists(mf_path):
        print(f"{RED}Lỗi: Không tìm thấy file {mf_path}. Hãy chạy huấn luyện BPR-MF trước.{RESET}")
        return
    mf_ckpt = np.load(mf_path)
    W_mf = mf_ckpt["user_embeddings"]  # (num_users, K)
    X_mf = mf_ckpt["item_embeddings"]  # (num_items, K)
    b_mf = mf_ckpt["item_biases"]      # (num_items,)

    # 5. Load LightGCN model and propagate embeddings
    print("5/5 Đang tải mô hình LightGCN và thực hiện lan truyền đặc trưng trên đồ thị...")
    sys.path.append(os.path.join(proj_root, "Model", "lightgcn-1M", "src"))
    try:
        from model import LightGCN
        from data import prepare_data
        from utils import get_device
        
        device = get_device()
        gcn_data = prepare_data(os.path.join(proj_root, "data"))
        gcn_ckpt_path = os.path.join(proj_root, "checkpoints", "best_lightgcn.pt")
        if not os.path.exists(gcn_ckpt_path):
            gcn_ckpt_path = os.path.join(proj_root, "checkpoints", "lightgcn.pt")
            
        gcn_ckpt = torch.load(gcn_ckpt_path, map_location=device)
        model_gcn = LightGCN(
            gcn_ckpt["num_users"],
            gcn_ckpt["num_items"],
            gcn_ckpt["embedding_dim"],
            gcn_ckpt["num_layers"]
        ).to(device)
        model_gcn.load_state_dict(gcn_ckpt["model_state"])
        model_gcn.eval()
        
        with torch.no_grad():
            user_emb, item_emb = model_gcn.propagate(gcn_data["norm_adj"].to(device))
            W_gcn = user_emb.cpu().numpy()
            X_gcn = item_emb.cpu().numpy()
    except Exception as e:
        print(f"{RED}Lỗi khi tải LightGCN: {e}. Vui lòng kiểm tra môi trường PyTorch.{RESET}")
        return

    print(f"{GREEN}{BOLD}Hệ thống đã sẵn sàng!.{RESET}")
    
    while True:
        print(f"\n{CYAN}{'-' * 110}{RESET}")
        user_input = input(f"{BOLD}Nhập User ID (Gốc: 1 - 6040, hoặc Mã hóa: 0 - 6039) [Nhập 'q' để thoát]: {RESET}").strip()
        
        if user_input.lower() in ["q", "exit", "quit"]:
            print(f"\n{GREEN}Thoát chương trình demo. {RESET}\n")
            break
            
        if not user_input.isdigit():
            print(f"{RED}Lỗi: Vui lòng nhập một số nguyên hợp lệ!{RESET}")
            continue
            
        val = int(user_input)
        
        # Determine user index (mapped)
        u = -1
        if val in user_ids:
            # Input was original UserID
            u = int(np.where(user_ids == val)[0][0])
        elif 0 <= val < num_users:
            # Input was mapped User ID
            u = val
        else:
            print(f"{RED}Lỗi: Không tìm thấy User ID '{val}' trong hệ thống.{RESET}")
            continue

        orig_uid = user_ids[u]
        seen_items = user_seen[u]
        
        print(f"\n{BOLD}=== THÔNG TIN NGƯỜI DÙNG ==={RESET}")
        print(f"* Chỉ mục đã mã hóa: {YELLOW}{u}{RESET} | ID gốc trong dataset: {YELLOW}{orig_uid}{RESET}")
        print(f"* Số lượng phim đã xem (tập Train): {YELLOW}{len(seen_items)}{RESET} phim")
        
        # Show movie history
        history_list = []
        for item_idx in list(seen_items)[:5]:
            orig_mid = item_ids[item_idx]
            title, genres = movie_info.get(orig_mid, (f"Movie {orig_mid}", "N/A"))
            history_list.append(f"{title} ({genres})")
        print(f"* Phim đã xem gần đây (mẫu 5 phim):")
        for i, h in enumerate(history_list):
            print(f"  {i+1}. {h}")
            
        # Validation movie (Ground Truth in Val set)
        val_item_idx = val_dict.get(u, -1)
        if val_item_idx != -1:
            orig_val_mid = item_ids[val_item_idx]
            val_title, val_genres = movie_info.get(orig_val_mid, (f"Movie {orig_val_mid}", "N/A"))
            print(f"* Phim thực tế làm mốc so sánh ({CYAN}Ground Truth - Val{RESET}):  {CYAN}{BOLD}{val_title}{RESET} ({val_genres})")
        else:
            print(f"* Phim so sánh ({CYAN}Ground Truth - Val{RESET}): Không tìm thấy trong tập Validation")
            val_title = "N/A"
            
        # Target movie (Ground Truth in Test set)
        test_item_idx = test_dict.get(u, -1)
        if test_item_idx != -1:
            orig_test_mid = item_ids[test_item_idx]
            test_title, test_genres = movie_info.get(orig_test_mid, (f"Movie {orig_test_mid}", "N/A"))
            print(f"* Phim thực tế sẽ xem tiếp theo ({GREEN}Ground Truth - Test{RESET}): {GREEN}{BOLD}{test_title}{RESET} ({test_genres})")
        else:
            print(f"* Phim thực tế ({GREEN}Ground Truth - Test{RESET}): Không tìm thấy trong tập Test")
            test_title = "N/A"

        # Predict with BPR-MF
        scores_mf = np.dot(W_mf[u], X_mf.T) + b_mf
        for seen_idx in seen_items:
            scores_mf[seen_idx] = -np.inf
        top20_mf = np.argsort(-scores_mf)[:20]

        # Predict with LightGCN
        scores_gcn = np.dot(W_gcn[u], X_gcn.T)
        for seen_idx in seen_items:
            scores_gcn[seen_idx] = -np.inf
        top20_gcn = np.argsort(-scores_gcn)[:20]

        # Check Hits
        mf_test_hit = test_item_idx in top20_mf
        mf_val_hit = val_item_idx in top20_mf
        
        gcn_test_hit = test_item_idx in top20_gcn
        gcn_val_hit = val_item_idx in top20_gcn

        # Build display table
        print(f"\n{BOLD}=== BẢNG SO SÁNH GỢI Ý TOP-20 PHIM CHƯA XEM ==={RESET}")
        print("+" + "-"*4 + "+" + "-"*52 + "+" + "-"*52 + "+")
        print(f"| {'STT':<2} | {'BPR-MF Gợi ý':<50} | {'LightGCN Gợi ý':<50} |")
        print("+" + "-"*4 + "+" + "-"*52 + "+" + "-"*52 + "+")

        for idx in range(20):
            # 1. Format MF column
            mf_item = top20_mf[idx]
            orig_mf_mid = item_ids[mf_item]
            mf_name, _ = movie_info.get(orig_mf_mid, (f"Movie {orig_mf_mid}", "N/A"))
            
            if mf_item == test_item_idx:
                mf_display_name = mf_name[:32]
                padding_needed = 50 - (len(mf_display_name) + 15)
                mf_cell = f"{GREEN}{BOLD}[★ HIT TEST ★] {mf_display_name}{' ' * padding_needed}{RESET}"
            elif mf_item == val_item_idx:
                mf_display_name = mf_name[:33]
                padding_needed = 50 - (len(mf_display_name) + 14)
                mf_cell = f"{CYAN}{BOLD}[✦ HIT VAL ✦] {mf_display_name}{' ' * padding_needed}{RESET}"
            else:
                mf_display_name = mf_name[:47] + "..." if len(mf_name) > 47 else mf_name.ljust(50)
                mf_cell = f"{WHITE}{mf_display_name}{RESET}"

            # 2. Format GCN column
            gcn_item = top20_gcn[idx]
            orig_gcn_mid = item_ids[gcn_item]
            gcn_name, _ = movie_info.get(orig_gcn_mid, (f"Movie {orig_gcn_mid}", "N/A"))
            
            if gcn_item == test_item_idx:
                gcn_display_name = gcn_name[:32]
                padding_needed = 50 - (len(gcn_display_name) + 15)
                gcn_cell = f"{GREEN}{BOLD}[★ HIT TEST ★] {gcn_display_name}{' ' * padding_needed}{RESET}"
            elif gcn_item == val_item_idx:
                gcn_display_name = gcn_name[:33]
                padding_needed = 50 - (len(gcn_display_name) + 14)
                gcn_cell = f"{CYAN}{BOLD}[✦ HIT VAL ✦] {gcn_display_name}{' ' * padding_needed}{RESET}"
            else:
                gcn_display_name = gcn_name[:47] + "..." if len(gcn_name) > 47 else gcn_name.ljust(50)
                gcn_cell = f"{WHITE}{gcn_display_name}{RESET}"

            # Pad values and print row
            stt = idx + 1
            print(f"| {stt:<2} | {mf_cell} | {gcn_cell} |")

        print("+" + "-"*4 + "+" + "-"*52 + "+" + "-"*52 + "+")

        # Summary results
        def get_status_str(test_hit, val_hit):
            if test_hit:
                return f"{GREEN}{BOLD}HIT TEST! (Trúng tập Test) ★{RESET}"
            elif val_hit:
                return f"{CYAN}{BOLD}HIT VAL! (Trúng tập Validation) ✦{RESET}"
            else:
                return f"{RED}MISS (Trượt){RESET}"

        print(f"\n{BOLD}KẾT QUẢ ĐÁNH GIÁ CHỈ SỐ HIT RATE (TOP-20):{RESET}")
        print(f"* Mô hình BPR-MF:   {get_status_str(mf_test_hit, mf_val_hit)}")
        print(f"* Mô hình LightGCN: {get_status_str(gcn_test_hit, gcn_val_hit)}")

if __name__ == "__main__":
    main()
