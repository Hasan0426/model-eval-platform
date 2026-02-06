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
        print("Database connected!")
    except Exception as e:
        print(f" Database connection failed: {e}")

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
        print(" MinIO connected and bucket ensured!")
    except Exception as e:
        print(f" MinIO connection failed: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "env": "docker-compose"}