# VidMatch-AI: AI-Powered Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**VidMatch-AI** là hệ thống gợi ý phim tiên tiến được xây dựng dựa trên công nghệ học sâu và học máy đồ thị. Dự án này triển khai và so sánh hai thuật toán gợi ý hàng đầu hiện nay trên tập dữ liệu chuẩn **MovieLens 1M**:
1.  **BPR-MF (Bayesian Personalized Ranking Matrix Factorization):** Mô hình phân rã ma trận được tối ưu hóa trực tiếp cho bài toán xếp hạng Top-K thông qua hàm mất mát học theo cặp (Pairwise Learning to Rank - BPR Loss) kết hợp độ lệch tự nhiên (Bias) của phim.
2.  **LightGCN (Light Graph Convolutional Networks):** Mô hình mạng nơ-ron đồ thị (GCN) thế hệ mới được thiết kế dành riêng cho tác vụ gợi ý bằng cách lược bỏ các phép biến đổi phi tuyến phức tạp (Feature Transformation & ReLU) để truyền tải thông tin sở thích (Collaborative Signal) qua đồ thị hiệu quả hơn.

Dự án được cấu hình chạy toàn bộ trên **NVIDIA GPU (CUDA)** thông qua môi trường **Docker** ảo hóa để đảm bảo tính đồng bộ, dễ dàng triển khai và không bị xung đột thư viện trên mọi hệ điều hành.

---

## Mục lục
- [1. Cấu trúc Thư mục Dự án](#1-cấu-trúc-thư-mục-dự-án)
- [2. Yêu cầu Hệ thống (Prerequisites)](#2-yêu-cầu-hệ-thống-prerequisites)
- [3. Hướng dẫn Cài đặt & Khởi chạy Docker](#3-hướng-dẫn-cài-đặt--khởi-chạy-docker)
- [4. Hướng dẫn Huấn luyện (Training)](#4-hướng-dẫn-huấn-luyện-training)
- [5. Chạy Demo Gợi ý Tương tác (Terminal Demo)](#5-chạy-demo-gợi-ý-tương-tác-terminal-demo)
- [6. Bảng kết quả Đánh giá Mô hình](#6-bảng-kết-quả-đánh-giá-mô-hình)

---

## 1. Cấu trúc Thư mục Dự án

```text
VidMatch-AI/
├── data/
│   └── EDA.ipynb                      # Notebook phân tích khám phá dữ liệu MovieLens (EDA)
├── datasets/
│   └── ml-1m/                         # Dataset MovieLens 1M (tự động tải & xử lý Leave-One-Out)
├── docker/
│   └── Dockerfile                     # Dockerfile cấu hình môi trường PyTorch GPU (CUDA 12.1)
├── checkpoints/                       # Thư mục lưu trọng số tốt nhất (.pt / .npz) và biểu đồ hội tụ (Loss Curves)
├── Model/
│   ├── lightgcn-1M/                   # Source code mô hình Mạng đồ thị LightGCN
│   │   ├── src/                       # Các module xử lý đồ thị, mô hình và huấn luyện
│   │   ├── config.yaml                # Tham số siêu cấu hình LightGCN (Epochs, Batch size, L2 Reg, v.v.)
│   │   └── README.md                  # Hướng dẫn chi tiết cho thư mục LightGCN
│   └── BPR-MF/                        # Source code mô hình phân rã ma trận
│       └── bpr_mf.py                  # Script Python huấn luyện và đánh giá BPR-MF
├── demo_terminal.py                   # Script chạy demo thực tế xem gợi ý Top-10 phim cho người dùng
├── docker-compose.yml                 # File docker-compose cấu hình ánh xạ cổng, GPU và volume
├── requirements.txt                   # Danh sách thư viện Python phụ trợ
└── README.md                          # Hướng dẫn tổng quan dự án
```

---

## 2. Yêu cầu Hệ thống (Prerequisites)

Để container Docker có thể truy cập và sử dụng GPU NVIDIA từ máy vật lý của bạn (Host), hệ thống cần đáp ứng các điều kiện sau:

### A. Cấu hình máy Host (Windows)
*   **Card màn hình:** NVIDIA GPU hỗ trợ CUDA (Khuyến nghị VRAM >= 4GB).
*   **Driver:** Cài đặt driver card đồ họa NVIDIA mới nhất trên Windows.
*   **WSL 2 (Windows Subsystem for Linux):** Đã kích hoạt và cài đặt (khuyên dùng Ubuntu distro).
*   **Docker Desktop:** Đã bật cấu hình **Use the WSL 2 based engine** trong phần cấu hình của Docker.

### B. Cấu hình NVIDIA Container Toolkit (Bắt buộc trên WSL2)
NVIDIA Container Toolkit cho phép Docker container sử dụng trực tiếp GPU của máy thật. Mở terminal của **WSL2 (Ubuntu)** và chạy các lệnh dưới đây để cài đặt:

```bash
# 1. Cấu hình khóa repository của NVIDIA
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Cập nhật gói và cài đặt toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. Đăng ký Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker

# 4. Khởi động lại Docker trong WSL
sudo systemctl restart docker
```
*Sau khi cài đặt xong, hãy **Restart** lại ứng dụng Docker Desktop trên Windows.*

---

## 3. Hướng dẫn Cài đặt & Khởi chạy Docker

Mở PowerShell hoặc CMD tại thư mục gốc của dự án trên Windows và thực hiện các bước sau:

### Bước 1: Build Image và Khởi động Container
Chạy lệnh sau để Docker tự động xây dựng môi trường và gắn kết mã nguồn:
```bash
docker compose up --build -d
```
*   Tham số `-d` giúp chạy container ngầm (background).
*   **Jupyter Notebook Server:** Sẽ được tự động bật và map ra cổng `http://127.0.0.1:8895`.

### Bước 2: Kiểm tra kết nối CUDA trong Docker
Chạy lệnh sau để xác nhận PyTorch bên trong Docker đã nhận diện card đồ họa:
```bash
docker compose exec app python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 4. Hướng dẫn Huấn luyện (Training)

Cả hai thuật toán đều được cấu hình chung một tập dữ liệu đã qua tiền xử lý, chia tách dựa trên cơ chế **Leave-One-Out** (Giữ lại item cuối cùng cho quá trình đánh giá khách quan). 

### A. Huấn luyện BPR-MF (Matrix Factorization)
*   **Lệnh huấn luyện (qua Docker):**
    ```bash
    docker compose exec app python Model/BPR-MF/bpr_mf.py
    ```
*   **Tuỳ chỉnh cấu hình (Ví dụ):**
    ```bash
    docker compose exec app python Model/BPR-MF/bpr_mf.py --k 20 --lam 0.01 --lr 0.05 --max-iter 40
    ```
*   **Đầu ra:** Mô hình sẽ tự động tính toán cả Train Loss và Validation Loss để vẽ đồ thị hội tụ vào thư mục `checkpoints/mf_loss.png` và lưu bộ trọng số tốt nhất vào `checkpoints/mf.pt` (hoặc `mf.npz`).

### B. Huấn luyện LightGCN
*   **Lệnh huấn luyện (qua Docker):**
    ```bash
    docker compose exec app python -m Model.lightgcn-1M.src.train --config Model/lightgcn-1M/config.yaml
    ```
*   **Tùy chỉnh cấu hình:** Tất cả siêu tham số (Epochs, L2 Reg, Embedding Dim, Batch Size) được lưu ở file `Model/lightgcn-1M/config.yaml`.
*   **Đầu ra:** Mô hình kích hoạt Early Stopping để ngăn Overfitting, tự động lưu `checkpoints/best_lightgcn.pt` và biểu đồ Loss tại `checkpoints/lightgcn_loss.png`.

---

## 5. Chạy Demo Gợi ý Tương tác (Terminal Demo)

Dự án cung cấp một giao diện mô phỏng trên terminal để thử nghiệm các gợi ý phim thực tế (Top-10) từ hai mô hình, đồng thời đánh dấu chính xác xem bộ phim gợi ý có xuất hiện (Hit) trong tập Validation/Test của người dùng hay không.

*   **Chạy trực tiếp (Nếu máy bạn có cài sẵn các thư viện):**
    ```bash
    python demo_terminal.py
    ```
*   **Hoặc chạy qua Docker:**
    ```bash
    docker compose exec app python demo_terminal.py
    ```
Bạn chỉ cần nhập ID người dùng (ví dụ: `15`, `256`) và hệ thống sẽ in ra danh sách lịch sử các phim người dùng đã xem, theo sau là gợi ý song song từ hai mô hình.

---

## 6. Bảng kết quả Đánh giá Mô hình

Dưới đây là bảng so sánh hiệu năng xếp hạng trên tập Test khách quan (không có rò rỉ dữ liệu) của hai mô hình trên MovieLens 1M. Đánh giá xếp hạng dùng cơ chế **Leave-One-Out** với các chỉ số **Recall@20** và **NDCG@20**.

| STT | Thuật toán gợi ý | K (Chiều không gian ẩn) | Epochs Hội tụ | Chỉ số Recall@20 | Chỉ số NDCG@20 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **BPR-MF** | 20 | 40 | 0.1209 | 0.0483 |
| 2 | **LightGCN** | 64 | 150 | **0.1384** | **0.0540** |

👉 *Nhận xét:* Kiến trúc truyền tin qua đồ thị đa tầng của **LightGCN** đã chứng minh hiệu quả vượt trội so với phân rã ma trận tuyến tính truyền thống (**BPR-MF**), tăng Recall@20 lên hơn **14.47%**.
