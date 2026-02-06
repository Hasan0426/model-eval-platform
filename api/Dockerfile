FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev libgomp1 && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
# (在实际生产中，建议生成 requirements.txt)
COPY pyproject.toml .
# `pip install .` 需要源码目录存在（setuptools packages = ["api", "worker", "core"]）
COPY api ./api
COPY worker ./worker
COPY core ./core
# 直接按 pyproject.toml 安装项目依赖（包含 xgboost / sklearn / mlflow 等）
RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir .

# 复制所有代码
COPY . .

# 设置 PYTHONPATH 包含根目录，这样能 import core
ENV PYTHONPATH=/app