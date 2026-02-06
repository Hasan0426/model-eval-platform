FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y gcc libpq-dev

# 安装 Python 依赖
# (在实际生产中，建议生成 requirements.txt)
COPY pyproject.toml .
# 这里用一个简单的 hack 提取依赖安装，或者你可以先手动 pip install
RUN pip install fastapi uvicorn pandas python-multipart celery redis sqlalchemy psycopg2-binary minio pydantic-settings

# 复制所有代码
COPY . .

# 设置 PYTHONPATH 包含根目录，这样能 import core
ENV PYTHONPATH=/app