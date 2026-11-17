"""
Tests for API v1 endpoints.
"""

import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    """Test health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "components" in data


def test_health_endpoint_components(client: TestClient):
    """Test health endpoint components."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    components = data["components"]
    assert "api" in components
    assert "database" in components
    assert "services" in components
