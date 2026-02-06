import io
from minio import Minio
from core.config import settings

# 初始化 MinIO 客户端
minio_client = Minio(
    settings.MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=False # 开发环境通常是 HTTP
)

def ensure_bucket_exists():
    if not minio_client.bucket_exists(settings.MINIO_BUCKET_NAME):
        minio_client.make_bucket(settings.MINIO_BUCKET_NAME)

def upload_file(dataset_id: str, filename: str, data: bytes) -> str:
    """
    上传文件到 MinIO，返回对象路径 (Key)
    """
    ensure_bucket_exists()
    
    # 构造唯一的存储路径
    object_name = f"{dataset_id}/{filename}"
    
    # 使用 BytesIO 包装二进制数据
    data_stream = io.BytesIO(data)
    
    minio_client.put_object(
        settings.MINIO_BUCKET_NAME,
        object_name,
        data_stream,
        length=len(data),
        content_type="text/csv"
    )
    
    return object_name

def download_file(object_name: str) -> io.BytesIO:
    """
    从 MinIO 下载文件到内存
    """
    response = minio_client.get_object(settings.MINIO_BUCKET_NAME, object_name)
    try:
        return io.BytesIO(response.read())
    finally:
        response.close()
        response.release_conn()