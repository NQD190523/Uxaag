"""
Tests for API v1 models.
"""

import pytest
from datetime import datetime

from app.schemas.pydantic_schemas import (
    BaseResponse, ErrorResponse, HealthResponse,
    UserRegistration, UserLogin, TokenResponse,
    DocumentExtractRequest, RAGRequest, ChatRequest
)


def test_base_response():
    """Test BaseResponse model."""
    response = BaseResponse(success=True, message="Test message")
    assert response.success is True
    assert response.message == "Test message"
    assert isinstance(response.timestamp, datetime)


def test_error_response():
    """Test ErrorResponse model."""
    response = ErrorResponse(error="Test error", message="Error message")
    assert response.success is False
    assert response.error == "Test error"
    assert response.message == "Error message"


def test_health_response():
    """Test HealthResponse model."""
    response = HealthResponse(
        status="healthy",
        version="1.0.0",
        message="Service is healthy",
        components={"api": "healthy"}
    )
    assert response.status == "healthy"
    assert response.version == "1.0.0"
    assert "api" in response.components


def test_user_registration():
    """Test UserRegistration model."""
    user_data = UserRegistration(
        username="testuser",
        email="test@example.com",
        password="password123",
        full_name="Test User"
    )
    assert user_data.username == "testuser"
    assert user_data.email == "test@example.com"
    assert user_data.password == "password123"
    assert user_data.full_name == "Test User"


def test_user_login():
    """Test UserLogin model."""
    login_data = UserLogin(username="testuser", password="password123")
    assert login_data.username == "testuser"
    assert login_data.password == "password123"


def test_token_response():
    """Test TokenResponse model."""
    token_data = TokenResponse(
        access_token="access_token",
        refresh_token="refresh_token",
        expires_in=3600
    )
    assert token_data.access_token == "access_token"
    assert token_data.refresh_token == "refresh_token"
    assert token_data.token_type == "bearer"
    assert token_data.expires_in == 3600
