from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import create_app


def test_metrics_endpoint_returns_request_count_metric():
    app = create_app()
    client = TestClient(app)

    # Prime middleware with a regular request to ensure metrics have samples
    health_resp = client.get("/health")
    assert health_resp.status_code == 200

    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text

    # Ensure our core metric families are exposed
    assert "request_count" in text
    assert "request_latency_seconds" in text