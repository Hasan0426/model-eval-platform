from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    """
    测试 /health 接口是否返回 200 OK
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "env": "docker-compose"}