这是一个激动人心的阶段！**Milestone C** 将标志着你的平台从“数据分析工具”正式升级为**“机器学习平台（MLOps Platform）”**。

我们将引入 **MLflow** 作为模型管理的核心，并集成 **LightGBM** 实现自动化训练。

我们将分 5 步完成这次升级：

1. **基础设施**：在 Docker 中添加 MLflow 服务。
2. **依赖管理**：添加 ML 相关的 Python 库。
3. **数据库**：新建 `TrainingJob` 表来追踪训练任务。
4. **核心逻辑**：编写训练 Baseline (LightGBM) 并记录到 MLflow 的代码。
5. **业务串联**：API 发起训练 -> Worker 执行 -> MLflow 记录。

---

### 第一步：基础设施 (Infra)

我们需要在 `docker-compose.yml` 中添加 MLflow 服务。MLflow 将使用我们现有的 Postgres 存元数据，用 MinIO 存模型文件。

#### 修改 `infra/docker-compose.yml`

在 `services:` 下添加 `mlflow` 服务，并更新 `api` and `worker` 的环境变量。

```yaml
services:
  # ... (db, minio, redis 保持不变) ...

  # [新增] MLflow Tracking Server
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.7.1
    ports:
      - "5000:5000"
    environment:
      # 让 MLflow 使用我们的 Postgres 存实验数据
      - MLFLOW_BACKEND_STORE_URI=postgresql://user:password@db:5432/model_eval_db
      # 让 MLflow 使用我们的 MinIO 存模型文件 (Artifacts)
      - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
      - AWS_ACCESS_KEY_ID=minioadmin
      - AWS_SECRET_ACCESS_KEY=minioadmin
    command: >
      mlflow server
      --backend-store-uri postgresql://user:password@db:5432/model_eval_db
      --default-artifact-root s3://mlflow/
      --host 0.0.0.0
    depends_on:
      - db
      - minio

  # [修改] API 服务 (增加 MLflow 环境变量)
  api:
    # ... build/ports/volumes 保持不变 ...
    environment:
      # ... 原有变量 ...
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
      - AWS_ACCESS_KEY_ID=minioadmin
      - AWS_SECRET_ACCESS_KEY=minioadmin

  # [修改] Worker 服务 (增加 MLflow 环境变量)
  worker:
    # ... build/command/volumes 保持不变 ...
    environment:
      # ... 原有变量 ...
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
      - AWS_ACCESS_KEY_ID=minioadmin
      - AWS_SECRET_ACCESS_KEY=minioadmin

```

---

### 第二步：依赖管理 (Dependencies)

#### 修改 `pyproject.toml`

添加机器学习和 MLflow 相关的库。

```toml
dependencies = [
    # ... 原有依赖 ...
    "mlflow",           # MLOps 核心
    "lightgbm",         # Baseline 模型
    "scikit-learn",     # 数据切分与预处理
    "boto3",            # MLflow 连接 MinIO 需要
    "numpy"             # 数值计算
]

```

---

### 第三步：数据库模型 (Models)

我们需要一张表来记录“谁在什么时候训练了什么模型，状态如何”。

#### 修改 `core/models.py`

在文件末尾追加 `TrainingJob` 类。

```python
# ... 原有 import ...

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(String, primary_key=True, index=True)
    dataset_id = Column(String, index=True) # 关联的数据集
    target_col = Column(String)             # 预测目标列
    task_type = Column(String)              # classification / regression
    model_name = Column(String)             # e.g., "LightGBM_Baseline"
    
    # 任务状态 (PENDING, RUNNING, COMPLETED, FAILED)
    status = Column(String, default="PENDING")
    
    # MLflow 关联信息
    mlflow_run_id = Column(String, nullable=True) # 对应 MLflow Run
    model_uri = Column(String, nullable=True)     # 模型存储位置 s3://...
    
    metrics = Column(JSON, nullable=True)         # 存 accuracy/f1 等
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

```

---

### 第四步：核心 ML 逻辑 (Core ML)

这是本里程碑的重头戏。我们需要一个模块来负责：读取数据 -> 清洗 -> 训练 -> 记录到 MLflow。

#### 创建 `core/ml.py`

```python
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from typing import Tuple, Dict, Any
import os

# 配置 MLflow 连接 (从环境变量读取)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

def train_lightgbm_baseline(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "classification",  # or "regression"
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[str, Dict[str, float]]:
    """
    训练 LightGBM Baseline 并记录到 MLflow
    返回: (run_id, metrics)
    """
    
    # 1. 数据预处理 (简易版)
    # 将 object 类型的列进行 LabelEncoding，因为 LightGBM 不支持字符串输入
    # (更复杂的系统会用 OneHot 或 TargetEncoding，这里 MVP 优先)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    label_encoders = {}
    for col in X.select_dtypes(include=['object', 'string']).columns:
        le = LabelEncoder()
        # 简单填充缺失值以防止报错，实际生产需精细处理
        X[col] = X[col].fillna("Unknown")
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # 划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2. 设置 MLflow 实验
    experiment_name = "baseline_training"
    mlflow.set_experiment(experiment_name)

    # 3. 开始 MLflow Run
    with mlflow.start_run() as run:
        # --- 记录参数 ---
        params = {
            "objective": "binary" if task_type == "classification" else "regression",
            "metric": "binary_logloss" if task_type == "classification" else "rmse",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "random_state": random_state
        }
        mlflow.log_params(params)
        
        # --- 训练模型 ---
        if task_type == "classification":
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            y_pred = clf.predict(X_val)
            
            # 计算指标
            acc = accuracy_score(y_val, y_pred)
            metrics = {"accuracy": acc}
        else:
            reg = lgb.LGBMRegressor(**params)
            reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            y_pred = reg.predict(X_val)
            
            # 计算指标
            mse = mean_squared_error(y_val, y_pred)
            metrics = {"mse": mse}

        # --- 记录指标 ---
        mlflow.log_metrics(metrics)

        # --- 记录模型 (保存到 MinIO) ---
        # 这一步会自动把模型序列化并上传到 s3://mlflow/
        mlflow.lightgbm.log_model(
            lgb_model=clf if task_type == "classification" else reg,
            artifact_path="model",
            registered_model_name="BaselineModel" # 注册到 Model Registry
        )
        
        print(f"✅ MLflow Run ID: {run.info.run_id}")
        return run.info.run_id, metrics

```

---

### 第五步：业务层 (API & Worker)

#### 1. 修改 `worker/tasks.py`

添加训练任务 `task_train_model`。

```python
# ... 之前的引用 ...
from core.models import TrainingJob # 新增引用
from core.ml import train_lightgbm_baseline # 新增引用

@celery_app.task(bind=True)
def task_train_model(self, job_id: str):
    db = SessionLocal()
    try:
        # 1. 获取任务信息
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            return "Job Not Found"
            
        job.status = "RUNNING"
        db.commit()

        # 2. 获取数据集信息
        dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
        if not dataset:
            raise ValueError("Dataset not found")

        # 3. 下载并读取数据 (复用 Milestone B 的逻辑)
        file_stream = storage.download_file(dataset.storage_path)
        df = pd.read_csv(file_stream)
        
        # 4. 执行训练 (调用 core/ml.py)
        # 这里假设目标列存在，实际需加校验
        run_id, metrics = train_lightgbm_baseline(
            df=df, 
            target_col=job.target_col, 
            task_type=job.task_type
        )
        
        # 5. 更新数据库
        job.mlflow_run_id = run_id
        job.metrics = metrics
        job.status = "COMPLETED"
        job.model_uri = f"runs:/{run_id}/model"
        db.commit()
        
        return "SUCCESS"

    except Exception as e:
        db.rollback()
        if job:
            job.status = "FAILED"
            job.error_message = str(e)
            db.commit()
        raise e
    finally:
        db.close()

```

#### 2. 修改 `api/main.py`

添加 API 接口。我们需要一个 Request Body 模型来接收参数。

```python
from pydantic import BaseModel

# ... 原有代码 ...
from core.models import TrainingJob
from worker.tasks import task_train_model # 引入新任务

# 定义请求体结构
class TrainRequest(BaseModel):
    dataset_id: str
    target_col: str
    task_type: str = "classification" # 默认分类

@app.post("/models/train")
async def train_model(
    req: TrainRequest,
    db: Session = Depends(get_db)
):
    # 1. 简单校验数据集是否存在
    dataset = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # 2. 创建训练任务记录
    job_id = str(uuid.uuid4())
    new_job = TrainingJob(
        id=job_id,
        dataset_id=req.dataset_id,
        target_col=req.target_col,
        task_type=req.task_type,
        model_name="LightGBM_Baseline",
        status="PENDING"
    )
    db.add(new_job)
    db.commit()

    # 3. 触发 Celery 任务
    task_train_model.delay(job_id)

    return {
        "job_id": job_id, 
        "status": "PENDING", 
        "mlflow_ui": "http://localhost:5000"
    }

# 可选：获取训练状态接口
@app.get("/models/jobs/{job_id}")
def get_training_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

```

---

### ✅ 运行与验收指南

由于我们修改了 `pyproject.toml` 和 `docker-compose.yml`，**必须**重新构建。

#### 1. 重新构建并启动

```bash
docker compose -f infra/docker-compose.yml down
docker compose -f infra/docker-compose.yml up --build -d

```

*等待约 1 分钟，确保 MLflow 服务完全启动。*

#### 2. 初始化 MLflow 的 S3 存储桶 (MinIO)

MLflow 需要一个叫 `mlflow` 的桶来存模型。虽然我们可以在代码里自动创建，但为了稳妥，我们手动检查一下：

1. 登录 MinIO (`http://localhost:9001`, minioadmin/minioadmin)。
2. 查看 Buckets，如果没有叫 `mlflow` 的桶，请点击 **Create Bucket** 创建一个名为 `mlflow` 的桶。

#### 3. 执行测试

1. **上传 CSV**：先用 POST `/datasets/upload` 上传你的测试数据（记得包含目标列，例如 `gender` 或 `department`）。拿到 `dataset_id`。
2. **发起训练**：
* POST `http://localhost:8000/models/train`
* Body:
```json
{
  "dataset_id": "你的数据集ID",
  "target_col": "gender",
  "task_type": "classification"
}

```




3. **等待结果**：调用 GET `/models/jobs/{job_id}` 直到状态变为 `COMPLETED`。

#### 4. 终极验收：查看 MLflow UI

1. 打开浏览器访问 **http://localhost:5000**。
2. 你应该能看到名为 `baseline_training` 的实验。
3. 点击进去，能看到刚刚的 Run。
4. 点击 Run，你可以看到：
* **Parameters**: `learning_rate`, `n_estimators` 等。
* **Metrics**: `accuracy`。
* **Artifacts**: 点击 Artifacts 标签，应该能看到 `model/model.pkl` 文件。这意味着模型已经成功保存到了 MinIO 中！



至此，你已经完成了一个**闭环的 MLOps 流程**。