FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV JUPYTER_CONFIG_DIR=/app

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy và cài đặt thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tạo symlink python3 để khớp cấu hình PyCharm
RUN mkdir -p /usr/local/bin && ln -sf /opt/conda/bin/python /usr/local/bin/python3

# Mặc định mở bash
CMD ["bash"]