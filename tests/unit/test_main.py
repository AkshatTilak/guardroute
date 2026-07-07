from fastapi.testclient import TestClient
from projects.guardroute.src.main import app

client = TestClient(app)


def test_health_check():
    """Verify that the health check endpoint returns 200 and correct status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["environment"] == "testing"
    assert "version" in data
