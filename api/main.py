from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid

from core.config import settings
from core.database import engine, Base, get_db
from core.models import Dataset
from core import storage
from worker.tasks import task_profile_dataset # 引入任务

from core.models import TrainingJob
from worker.tasks import task_train_model # 引入新任务
from pydantic import BaseModel

# 自动创建表 (生产环境通常用 Alembic，MVP 用这个就行)
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.PROJECT_NAME)


# 定义请求体结构
class TrainRequest(BaseModel):
    dataset_id: str
    target_col: str
    task_type: str = "classification" # 默认分类

@app.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV supported")

    # 1. 生成 ID
    dataset_id = str(uuid.uuid4())
    content = await file.read()
    file_size = len(content)

    # 2. 上传 MinIO
    try:
        minio_path = storage.upload_file(dataset_id, file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage upload failed: {str(e)}")

    # 3. 写入 Postgres (状态: PENDING)
    new_dataset = Dataset(
        id=dataset_id,
        filename=file.filename,
        storage_path=minio_path,
        file_size=file_size,
        status="PENDING"
    )
    db.add(new_dataset)
    db.commit()
    db.refresh(new_dataset)

    # 4. 触发异步 Celery 任务
    task_profile_dataset.delay(dataset_id)

    return {
        "dataset_id": dataset_id,
        "status": "PENDING",
        "message": "Upload successful. Profiling started."
    }

@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    # 直接查询数据库
    ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds

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
        model_name="XGBoost_Baseline",
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