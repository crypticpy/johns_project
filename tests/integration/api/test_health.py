from fastapi.testclient import TestClient

from api.main import create_app


def test_health_endpoint_returns_ok_status():
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("status") == "ok"
    assert isinstance(payload.get("version"), str) and payload.get("version")