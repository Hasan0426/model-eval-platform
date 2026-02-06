好的，我们现在暂停业务功能的开发，专心把**地基（Milestone A）**打牢。

目前的痛点是：代码都在一个文件夹里，没有数据库，没有异步队列，全是本地文件操作，不具备“可扩展平台”的形态。

我们将按照你提供的大纲，把现有的代码拆解重组，并引入 Docker 环境。

---

### 第一步：重构目录结构 (Monorepo 改造)

我们需要把现在的扁平结构改成标准的工程结构。

**目标结构：**

```text
my-platform/
├── api/                 # Web 服务端
│   ├── Dockerfile
│   ├── main.py          # 入口
│   └── routers/         # 路由
├── worker/              # 异步任务执行器
│   ├── Dockerfile
│   ├── celery_app.py    # Celery 配置
│   └── tasks.py         # 任务逻辑
├── core/                # 公共核心库 (API 和 Worker 共用)
│   ├── config.py        # 环境变量配置
│   ├── profile.py       # (原 dataset_profile.py)
│   └── storage.py       # (原 dataset_store.py)
├── infra/               # 基础设施配置
│   └── docker-compose.yml
├── pyproject.toml       # 依赖与工具配置 (Ruff/Black)
└── .pre-commit-config.yaml # 代码提交检查

```

#### 1. 移动文件与调整引用

你需要手动移动文件，并修改 `import` 路径。

* 把 `app/services/dataset_profile.py` 移动到 `core/profile.py`。
* 把 `app/services/dataset_store.py` 移动到 `core/storage.py`。
* 把 `app/main.py` 移动到 `api/main.py`。

---

### 第二步：基础设施 (Docker Compose)

这是 Milestone A 的核心验收标准：**一键启动所有服务**。

在 `infra/` 目录下创建 `docker-compose.yml`。我们将启动：

1. **Postgres** (元数据存储)
2. **MinIO** (对象存储，S3 兼容，替代本地文件系统)
3. **Redis** (消息队列，给 Worker 用)
4. **API** (你的 FastAPI)
5. **Worker** (你的 Celery)

```yaml
# infra/docker-compose.yml
version: '3.8'

services:
  # 1. 数据库
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: model_eval_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # 2. 对象存储 (S3 Compatible)
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"  # API 端口
      - "9001:9001"  # 控制台端口
    volumes:
      - minio_data:/data

  # 3. 消息队列
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  # 4. API 服务
  api:
    build:
      context: ..
      dockerfile: api/Dockerfile
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    volumes:
      - ../:/app
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/model_eval_db
      - MINIO_ENDPOINT=minio:9000
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - minio
      - redis

  # 5. Worker 服务
  worker:
    build:
      context: ..
      dockerfile: worker/Dockerfile
    # 启动 celery worker
    command: celery -A worker.celery_app worker --loglevel=info
    volumes:
      - ../:/app
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/model_eval_db
      - MINIO_ENDPOINT=minio:9000
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - minio
      - redis

volumes:
  postgres_data:
  minio_data:

```

---

### 第三步：配置管理与依赖 (Core)

为了让代码能读到 Docker 里的环境变量，我们需要一个配置中心。

**1. 创建 `core/config.py**`

```python
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Model Eval Platform"
    
    # 数据库
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/model_eval_db")
    
    # MinIO
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ROOT_USER", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
    MINIO_BUCKET_NAME: str = "datasets"
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

settings = Settings()

```

**2. 创建 `pyproject.toml` (代码规范与依赖)**

这是 Python 现代化项目的标配，替代散乱的配置文件。

```toml
[project]
name = "model-eval-platform"
version = "0.1.0"
dependencies = [
    "fastapi",
    "uvicorn",
    "pandas",
    "python-multipart",
    "celery",
    "redis",
    "sqlalchemy",
    "psycopg2-binary",
    "minio",
    "pydantic-settings"
]

[tool.ruff]
# 代码检查工具配置
line-length = 120
select = ["E", "F", "I"] # 检查错误、格式、Import顺序

```

---

### 第四步：实现 Dockerfile

API 和 Worker 可以共用基础镜像，但在构建时指定不同的入口。

**`api/Dockerfile` (和 `worker/Dockerfile` 也可以复用同一个)**

```dockerfile
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

```

---

### 第五步：验证连接 (Health Check)

按照验收标准，我们需要确保 API 启动时能连上 Postgres 和 MinIO。

修改 `api/main.py`，增加启动时的探测逻辑：

```python
from fastapi import FastAPI
from sqlalchemy import create_engine, text
from minio import Minio
from core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

@app.on_event("startup")
def startup_check():
    # 1. 检查 Postgres 连接
    try:
        engine = create_engine(settings.DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Database connected!")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")

    # 2. 检查 MinIO 连接
    try:
        client = Minio(
            settings.MINIO_ENDPOINT.replace("http://", ""), # minio库不需要http前缀
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False
        )
        if not client.bucket_exists(settings.MINIO_BUCKET_NAME):
            client.make_bucket(settings.MINIO_BUCKET_NAME)
        print("✅ MinIO connected and bucket ensured!")
    except Exception as e:
        print(f"❌ MinIO connection failed: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "env": "docker-compose"}

```

---

### 第六步：运行与验收

现在，你的工程已经是“正规军”了。

1. **启动平台：**
在项目根目录下执行：
```bash
docker-compose -f infra/docker-compose.yml up --build

```


2. **验收：**
* 观察终端日志，是否出现 `✅ Database connected!` 和 `✅ MinIO connected!`。
* 打开浏览器访问 `http://localhost:8000/docs`。
* 打开 MinIO 控制台 `http://localhost:9001` (账号密码 minioadmin/minioadmin)，看是否能登录。



### 接下来要做什么 (Next Step)

虽然环境通了，但你的代码（Profile/Upload）还在用**本地文件系统**。
下一步，我将帮你把 `core/storage.py` 从读写本地文件，改造为**读写 MinIO 和 数据库**，正式完成 Milestone B 的所有内容。

**你需要我现在给出重写后的 `core/storage.py` 代码吗？**