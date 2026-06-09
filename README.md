# VidMatch-AI: AI-Powered Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**VidMatch-AI** là hệ thống gợi ý phim tiên tiến được xây dựng dựa trên công nghệ học sâu và học máy đồ thị. Dự án này triển khai và so sánh hai thuật toán gợi ý hàng đầu hiện nay trên tập dữ liệu chuẩn **MovieLens 1M**:
1.  **Matrix Factorization (MF) with Bias:** Mô hình phân rã ma trận tối ưu hóa bằng thuật toán Batch Gradient Descent kết hợp độ lệch (User/Item Biases).
2.  **LightGCN (Light Graph Convolutional Networks):** Mô hình mạng nơ-ron đồ thị (GCN) thế hệ mới được tối ưu hóa cho tác vụ gợi ý bằng cách lược bỏ các phép biến đổi phi tuyến phức tạp.

Dự án được cấu hình chạy toàn bộ trên **NVIDIA GPU (CUDA)** thông qua môi trường **Docker** ảo hóa để đảm bảo tính đồng bộ, dễ dàng triển khai và không bị xung đột thư viện trên mọi hệ điều hành.

---

## Mục lục
- [1. Cấu trúc Thư mục Dự án](#1-cấu-trúc-thư-mục-dự-án)
- [2. Yêu cầu Hệ thống (Prerequisites)](#2-yêu-cầu-hệ-thống-prerequisites)
- [3. Hướng dẫn Cài đặt & Khởi chạy Docker](#3-hướng-dẫn-cài-đặt--khởi-chạy-docker)
- [4. Cấu hình với IDE (PyCharm / IntelliJ / VS Code)](#4-cấu-hình-với-ide-pycharm--intellij--vs-code)
- [5. Hướng dẫn Huấn luyện & Đánh giá](#5-hướng-dẫn-huấn-luyện--đánh-giá)
- [6. Bảng kết quả Đánh giá Mô hình](#6-bảng-kết-quả-đánh-giá-mô-hình)

---

## 1. Cấu trúc Thư mục Dự án

```text
VidMatch-AI/
├── data/
│   └── EDA.ipynb                      # Notebook phân tích khám phá dữ liệu MovieLens (EDA)
├── datasets/
│   └── ml-1m/                         # Dataset MovieLens 1M (tự động tải nếu thiếu)
├── docker/
│   └── Dockerfile                     # Dockerfile cấu hình môi trường PyTorch GPU (CUDA 12.1)
├── Model/
│   ├── lightgcn-1M/                   # Source code mô hình học sâu đồ thị LightGCN cho ML-1M
│   │   ├── src/                       # Các module xử lý dữ liệu, mô hình và huấn luyện
│   │   ├── config.yaml                # Tham số cấu hình LightGCN
│   │   └── README.md                  # Hướng dẫn chi tiết cho LightGCN
│   └── MF/                            # Source code mô hình phân rã ma trận
│       └── mf_with_bias.py            # Script Python triển khai, chạy và đánh giá MF (tự động tải data)
├── docker-compose.yml                 # File docker-compose cấu hình ánh xạ cổng, GPU và volume
├── requirements.txt                   # Danh sách thư viện Python phụ trợ
└── README.md                          # Hướng dẫn tổng quan dự án
```

---

## 2. Yêu cầu Hệ thống (Prerequisites)

Để container Docker có thể truy cập và sử dụng GPU NVIDIA từ máy vật lý của bạn (Host), hệ thống cần đáp ứng các điều kiện sau:

### A. Cấu hình máy Host (Windows)
*   **Card màn hình:** NVIDIA GPU hỗ trợ CUDA.
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
Chạy lệnh sau để Docker tự động xây dựng môi trường phát triển:
```bash
docker-compose up --build -d
```
*   Tham số `-d` giúp chạy container ngầm (background).
*   **Jupyter Notebook Server:** Được mở tại cổng máy thật: `http://127.0.0.1:8895`.

### Bước 2: Kiểm tra CUDA GPU trong Docker
Chạy lệnh sau để xác nhận rằng PyTorch bên trong Docker đã nhận diện card đồ họa thành công:
```bash
docker-compose exec app python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
*Kết quả hiển thị `CUDA Available: True` kèm tên card đồ họa NVIDIA của bạn là cấu hình GPU thành công.*

---

## 4. Cấu hình với IDE (PyCharm / IntelliJ / VS Code)

Để lập trình và gỡ lỗi trực tiếp trong môi trường Docker thông qua IDE:

### A. Cấu hình Python Interpreter
1. Vào **Settings / Preferences** (`Ctrl + Alt + S`) -> **Project: VidMatch-AI** -> **Python Interpreter**.
2. Chọn biểu tượng bánh răng/Add Interpreter -> Chọn **Add New Interpreter > On Docker Compose...**
3. Cấu hình đường dẫn:
   * **Configuration files:** Chọn file `docker-compose.yml` của dự án.
   * **Service:** Chọn service `app`.
4. Nhấn **Next / Finish**. IDE sẽ tự động ánh xạ môi trường và nhận diện đầy đủ thư viện PyTorch CUDA trong Docker.

### B. Chạy Jupyter Notebook (`.ipynb`)
Mở notebook (ví dụ: `data/EDA.ipynb`):
1. Đảm bảo chọn tùy chọn **`IDE-Managed Server (Auto)`** trên thanh công cụ của Notebook ở góc trên bên phải.
2. Nhấn nút **Run Cell** (hình tam giác màu xanh) bên cạnh code để chạy.
3. *Lưu ý:* Dự án đã được cấu hình sẵn các file `jupyter_server_config.py` ở thư mục gốc để tự động điều phối IP `0.0.0.0`, đảm bảo PyCharm kết nối trực tiếp vào Docker mà không gặp bất cứ lỗi cổng (`NoHttpResponseException`) nào.

---

## 5. Hướng dẫn Huấn luyện & Đánh giá

### A. Huấn luyện mô hình Phân rã ma trận (Matrix Factorization)
*   **Chạy trực tiếp từ dòng lệnh Docker:**
    ```bash
    docker-compose exec app python -m Model.MF.mf_with_bias
    ```
*   **Tùy biến tham số (CLI Options):**
    ```bash
    docker-compose exec app python -m Model.MF.mf_with_bias --k 20 --lam 0.05 --lr 0.5 --max-iter 30
    ```
*   **Đặc điểm:** Code tính toán lỗi và đánh giá RMSE đã được tối ưu hóa vector hóa bằng NumPy, cho phép huấn luyện cực kỳ nhanh chóng. Sau khi chạy xong, script sẽ tự động lưu biểu đồ hội tụ của Loss và RMSE tại `checkpoints/mf_convergence_1m.png`.

### B. Huấn luyện mô hình Mạng Đồ Thị (LightGCN)
*   **Chạy trực tiếp từ dòng lệnh Docker:**
    ```bash
    docker-compose exec app python -m Model.lightgcn-1M.src.train --config Model/lightgcn-1M/config.yaml
    ```
*   **Đặc điểm:** Tự động tối ưu hoá siêu tham số theo cấu hình trong file `Model/lightgcn-1M/config.yaml`. Kết quả checkpoint mô hình và biểu đồ đánh giá Metrics (Recall@20, NDCG@20) sẽ được lưu lại trong thư mục `checkpoints/`.

---

## 6. Bảng kết quả Đánh giá Mô hình

Dưới đây là bảng so sánh hiệu năng của các mô hình gợi ý đã triển khai trên tập dữ liệu MovieLens 1M:

| STT | Mô hình gợi ý | Dataset thử nghiệm | K (Nhân tử ẩn / Chiều nhúng) | Epochs / Max Iter | Normalization / Phân tách | Chỉ số Đánh giá |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **Item-based MF with Bias** | MovieLens 1M | 20 | 30 | Item-based Mean / Split ngẫu nhiên (80/20) | Test RMSE: ~1.4372 (với Item-based normalization) |
| 2 | **LightGCN (Graph Learning)**| MovieLens 1M | 64 | 15 | Phân tách Leave-one-out | Recall@20: ~0.0833 (NDCG@20: ~0.0338) |

*(Chỉ số chính xác cụ thể có thể dao động nhỏ tùy thuộc vào tham số khởi tạo ngẫu nhiên).*
