from sqlalchemy import Column, String, Integer, DateTime, JSON, Text
from sqlalchemy.sql import func
from core.database import Base

class Dataset(Base):
    __tablename__ = "app_datasets"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)  # MinIO 中的路径
    
    # 状态: PENDING (上传中/等待中), PROCESSING (分析中), COMPLETED (完成), FAILED (失败)
    status = Column(String, default="PENDING", index=True)
    
    # 元数据 (行数、列数、大小)
    row_count = Column(Integer, nullable=True)
    col_count = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)

    # Profiling 结果 (存为 JSON)
    schema_info = Column(JSON, nullable=True)      # 字段类型
    missing_stats = Column(JSON, nullable=True)    # 缺失值
    categorical_stats = Column(JSON, nullable=True)# 类别分布
    numerical_stats = Column(JSON, nullable=True)  # 数值分布 (新增)
    preview = Column(JSON, nullable=True)          # 前 N 行预览

    error_message = Column(Text, nullable=True)    # 如果失败，存报错信息

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    


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