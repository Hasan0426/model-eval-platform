import os
from celery import Celery
from core.config import settings

# 1. 初始化 Celery 实例
# "model_eval_worker" 是这个 Worker 的名字
celery_app = Celery(
    "model_eval_worker",
    broker=settings.REDIS_URL,   # 消息队列地址 (Redis)
    backend=settings.REDIS_URL   # 结果存储地址 (Redis)
)

# 2. 配置 Celery
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

# 3. 自动发现任务
# 告诉 Celery 去 worker.tasks 这个模块里找被 @task 装饰的函数
celery_app.autodiscover_tasks(["worker.tasks"])