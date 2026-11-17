# RAG Chat Room Web microservices - API Specification

## API Gateway (Port 8000)

### Health Check
```
GET /health
Response: {
  "status": "healthy",
  "service": "api-gateway",
  "version": "1.0.0",
  "services": {
    "auth": {...},
    "chat": {...},
    "document": {...},
    ...
  }
}
```

### Root Information
```
GET /
Response: {
  "message": "RAG Chat Room Web API Gateway",
  "version": "1.0.0",
  "services": {
    "auth": "http://localhost:8006/api/v1",
    "chat": "http://localhost:8005/api/v1",
    ...
  }
}
```

---

## Auth Service (Port 8006)

### Base URL
`http://localhost:8006/api/v1`

### Health Check
```
GET /health
Response: {
  "status": "healthy",
  "service": "auth-service",
  "version": "1.0.0"
}
```

### Register User
```
POST /auth/register
Content-Type: application/json

Request Body:
{
  "email": "user@example.com",
  "password": "securepassword123",
  "full_name": "John Doe"  // optional
}

Success Response (201):
{
  "user_id": "12345",
  "username": null,
  "email": "user@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "created_at": "2024-01-08T10:30:00.000Z",
  "last_login": null
}

Error Response (400):
{
  "detail": "Email already exists"
}
```

### Login User
```
POST /auth/login
Content-Type: application/json

Request Body:
{
  "email": "user@example.com",
  "password": "securepassword123"
}

Success Response (200):
{
  "success": true,
  "message": "Authentication successful",
  "user": {
    "id": 12345,
    "username": "johndoe",
    "email": "user@example.com",
    "full_name": "John Doe"
  },
  "tokens": {
    "access_token": "eyJhbGc...",
    "refresh_token": "eyJhbGc...",
    "token_type": "bearer",
    "expires_in": 1800
  }
}

Error Response (401):
{
  "detail": "Invalid credentials"
}
```

### Verify Token
```
GET /auth/verify/{token}

Success Response (200):
{
  "success": true,
  "user": {
    "user_id": 12345,
    "username": "johndoe",
    "email": "user@example.com",
    "is_active": true
  }
}

Error Response (401):
{
  "detail": "Invalid or expired token"
}
```

---

## Chat Service (Port 8005)

### Base URL
`http://localhost:8005/api/v1`

### Health Check
```
GET /health
Response: {
  "status": "healthy",
  "service": "chat-service",
  "version": "1.0.0"
}
```

### Send Message (Placeholder)
```
POST /chat/messages
Content-Type: application/json

Request Body:
{
  "message": "Hello",
  "user_id": 12345,
  "conversation_id": "conv_123"
}

Response:
{
  "success": true,
  "message": "Message received",
  "data": {...}
}
```

### Create Chat Room (Placeholder)
```
POST /chat/rooms
Content-Type: application/json

Request Body:
{
  "name": "General Chat",
  "user_id": 12345
}

Response:
{
  "success": true,
  "message": "Chat room created",
  "room_id": "room_123",
  "name": "General Chat"
}
```

### Get User Rooms (Placeholder)
```
GET /chat/rooms/{user_id}

Response:
{
  "success": true,
  "rooms": []
}
```

---

## Document Service (Port 8001)

### Base URL
`http://localhost:8001/api/v1`

### Health Check
```
GET /health
Response: {
  "status": "healthy",
  "service": "document-service",
  "version": "1.0.0"
}
```

### Upload Document (Placeholder)
```
POST /documents/upload
Content-Type: multipart/form-data

Response:
{
  "success": true,
  "message": "Document service is operational"
}
```

---

## RAG Service (Port 8002)

### Base URL
`http://localhost:8002/api/v1`

### Health Check
```
GET /health
Response: {
  "status": "healthy",
  "service": "rag-service",
  "version": "1.0.0"
}
```

### Query RAG (Placeholder)
```
POST /rag/query
Content-Type: application/json

Response:
{
  "success": true,
  "message": "RAG service is operational"
}
```

---

## Vector Store Service (Port 8003)

### Base URL
`http://localhost:8003/api/v1`

### Health Check
```
GET /health
Response: {
  "status": "healthy",
  "service": "vector-store-service",
  "version": "1.0.0"
}
```

### Store Vectors (Placeholder)
```
POST /vectors/store
Content-Type: application/json

Response:
{
  "success": true,
  "message": "Vector Store service is operational"
}
```

---

## Web Crawler Service (Port 8004)

### Base URL
`http://localhost:8004/api/v1`

### Health Check
```
GET /health
Response: {
  "status": "healthy",
  "service": "web-crawler-service",
  "version": "1.0.0"
}
```

### Crawl URL (Placeholder)
```
POST /crawl
Content-Type: application/json

Response:
{
  "success": true,
  "message": "Web Crawler service is operational"
}
```

---

## Standard Error Responses

All services follow this error format:

### 400 Bad Request
```json
{
  "detail": "Description of what went wrong"
}
```

### 401 Unauthorized
```json
{
  "detail": "Authentication required or invalid credentials"
}
```

### 403 Forbidden
```json
{
  "detail": "You don't have permission to access this resource"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "An unexpected error occurred"
}
```

---

## Authentication

Services that require authentication use Bearer tokens in the Authorization header:

```
Authorization: Bearer <access_token>
```

Example:
```bash
curl -H "Authorization: Bearer eyJhbGc..." http://localhost:8006/api/v1/protected
```

---

## Rate Limiting

Currently not implemented. Can be added using:
- `slowapi` library for FastAPI
- Redis for distributed rate limiting

---

## CORS

All services have CORS enabled with the following settings:
- **allow_origins**: `["*"]`
- **allow_credentials**: `true`
- **allow_methods**: `["*"]`
- **allow_headers**: `["*"]`

This can be restricted in production by modifying the CORS configuration in each service's `main.py`.

---

## Pagination (Future)

Services that return lists should support pagination:

```
GET /resource?skip=0&limit=10

Response:
{
  "items": [...],
  "total": 100,
  "skip": 0,
  "limit": 10
}
```

---

## Version Strategy

APIs use URL versioning: `/api/v1/`

Future versions will be `/api/v2/`, etc.

---

## WebSocket Support

Chat Service can support WebSocket connections for real-time messaging:

```
WS ws://localhost:8005/api/v1/chat/ws/{conversation_id}
```

---

## Service Discovery

Services can be discovered via the API Gateway's health endpoint or through environment configuration.

For containerized environments, use service names:
- `auth-service:8006`
- `chat-service:8005`
- etc.
