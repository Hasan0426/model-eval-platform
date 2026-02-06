import os
import pandas as pd
import traceback
from worker.celery_app import celery_app
from core.database import SessionLocal
from core.models import Dataset
from core import storage
from core.profile import (
    ProfileConfig, infer_schema, validate_schema, 
    missing_stats, categorical_distributions, numerical_stats, get_preview
)
from core.models import TrainingJob # 新增引用
# 新增引用
from core.ml import train_transformer_model

from core.storage import download_file
@celery_app.task(bind=True)
def task_profile_dataset(self, dataset_id: str):
    # Worker 需要自己创建 DB 会话
    db = SessionLocal()
    try:
        # 1. 获取任务记录
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            return "Dataset Not Found"
            
        # 更新状态为 PROCESSING
        dataset.status = "PROCESSING"
        db.commit()

        # 2. 从 MinIO 下载文件
        file_stream = storage.download_file(dataset.storage_path)
        df = pd.read_csv(file_stream)
        
        # 3. 执行 Profiling
        cfg = ProfileConfig()
        
        schema = infer_schema(df, cfg)
        validation = validate_schema(df)
        missing = missing_stats(df)
        cat_dist = categorical_distributions(df, cfg)
        num_stats = numerical_stats(df) # 新增
        preview = get_preview(df)       # 新增
        
        # 4. 更新数据库结果
        dataset.row_count = int(df.shape[0])
        dataset.col_count = int(df.shape[1])
        dataset.schema_info = schema
        dataset.schema_check = validation # 需在 Model 里加个字段，或者存 schema_info 里
        dataset.missing_stats = missing
        dataset.categorical_stats = cat_dist
        dataset.numerical_stats = num_stats
        dataset.preview = preview
        
        dataset.status = "COMPLETED"
        db.commit()
        return "SUCCESS"

    except Exception as e:
        db.rollback()
        # 记录错误信息
        if dataset:
            dataset.status = "FAILED"
            dataset.error_message = f"{str(e)}\n{traceback.format_exc()}"
            db.commit()
        raise e
    finally:
        db.close()
# worker/tasks.py

@celery_app.task(bind=True)
def task_train_model(self, job_id: str):
    db = SessionLocal() # 移除 :Session 类型注解
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    
    # 变量改个名，明确它是流，不是路径
    file_stream = None 
    
    try:
        if not job:
            print(f"❌ Job {job_id} not found.")
            return

        # 更新状态为 RUNNING
        job.status = "RUNNING"
        db.commit()

        # 1. 下载数据
        dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {job.dataset_id} not found")

        print(f"📥 Downloading dataset from: {dataset.storage_path}")
        # 🔥 这里返回的是 BytesIO 对象
        file_stream = download_file(dataset.storage_path)
        
        # 直接读取流
        df = pd.read_csv(file_stream)

        # 2. 调用 Transformer 训练函数
        print(f"🚀 Starting Transformer training for Job {job_id}...")
        run_id, metrics = train_transformer_model(
            df=df,
            target_col=job.target_col,
            task_type=job.task_type
        )

        # 3. 更新数据库
        job.mlflow_run_id = run_id
        job.metrics = metrics
        job.status = "COMPLETED"
        db.commit()
        print(f"✅ Job {job_id} completed successfully.")

    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        traceback.print_exc()
        try:
            job.status = "FAILED"
            job.error_message = str(e)
            db.commit()
        except:
            db.rollback()
    finally:
        db.close()
        # 🔥🔥🔥 删除了 os.remove 代码块 🔥🔥🔥
        # 因为内存对象不需要（也不能）用 os.remove 删除