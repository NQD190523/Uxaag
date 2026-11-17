# RAG Chat Room Web

A comprehensive FastAPI application for collaborative chat rooms with Retrieval-Augmented Generation (RAG), document processing, link indexing, and real-time messaging.

## 🏗️ Project Structure

```
rag-chat-room-web/
├── main.py                    # Entry point for your FastAPI application
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker containerfile
├── README.md                  # Project documentation
├── .gitignore                 # Define what to ignore during version control
│
├── app/                       # Application directory
│   ├── __init__.py            # Initialize the app package
│   ├── api/                   # API endpoints
│   │   ├── __init__.py        # Initialize the api package
│   │   ├── v1/                # Versioned API endpoints
│   │   │   ├── __init__.py
│   │   │   ├── endpoints.py   # Define API routes and handlers
│   │   │   └── dependencies.py # Dependency injection
│   ├── core/                  # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py          # Application configurations
│   │   ├── logging.py         # Logging utilities
│   │   ├── database_manager.py # Database management
│   │   └── schemas.py         # Shared Pydantic schemas
│   ├── db/                    # Database related files
│   │   ├── __init__.py
│   │   ├── models.py          # Database models
│   │   └── migrations/        # Database migrations
│   ├── services/              # Business logic layer
│   │   ├── __init__.py
│   │   ├── auth_service.py    # Authentication service
│   │   ├── document_service.py # Document processing service
│   │   ├── rag_service.py     # RAG service
│   │   ├── vector_service.py  # Vector store service
│   │   ├── web_crawler_service.py # Web crawling service
│   │   └── chat_service.py    # Chat service
│   ├── repositories/          # Database logic layer
│   │   ├── __init__.py
│   │   └── base.py            # Base repository classes
│   ├── utils/                 # Utility functions
│   │   └── __init__.py
│   └── schemas/               # Pydantic schemas
│       ├── __init__.py
│       ├── pydantic_schemas.py # Main schemas
│       └── auth_schemas.py    # Authentication schemas
│
├── tests/                     # Test directory
│   ├── __init__.py
│   ├── conftest.py            # Test fixtures
│   ├── test_main.py           # Tests for main module
│   └── test_api/              # API endpoint tests
│       ├── __init__.py
│       ├── test_v1/           # Versioned API endpoint tests
│       │   ├── __init__.py
│       │   ├── test_endpoints.py
│       │   └── test_models.py
│       └── ...
│
├── configs/                   # Configuration files
│   ├── config.yaml           # Main configuration
│   ├── microservices.yaml    # Microservices configuration
│   └── source.yaml           # Source configuration
│
├── data/                     # Data directory
│   ├── vector_store/         # Vector database storage
│   └── logs/                 # Application logs
│
└── services/                 # Legacy microservices (for reference)
    ├── shared/               # Shared components
    ├── auth-service/         # Authentication microservice
    ├── document-service/     # Document microservice
    ├── rag-service/          # RAG microservice
    ├── vector-store-service/ # Vector store microservice
    ├── web-crawler-service/  # Web crawler microservice
    ├── chat-service/         # Chat microservice
    └── api-gateway/          # API Gateway microservice
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Poetry (recommended) or pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd rag-chat-room-web
   ```

2. **Install dependencies with Poetry (recommended):**
   ```bash
   poetry install
   ```

   **Or with pip:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env with your configuration
   OPENAI_API_KEY=your-openai-api-key-here
   JWT_SECRET_KEY=your-super-secret-jwt-key-here
   ```

4. **Run the application:**
   
   **With Poetry:**
   ```bash
   poetry run python main.py
   ```
   
   **Or with pip:**
   ```bash
   python main.py
   ```

5. **Access the application:**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/api/v1/health

### Docker

1. **Build and run with Docker:**
   ```bash
   docker build -t rag-chat-room-web .
   docker run -p 8000:8000 rag-chat-room-web
   ```

## 🔧 Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your-openai-api-key-here

# Optional
DATABASE_URL=sqlite:///./data/rag_chat_room_web.db
JWT_SECRET_KEY=your-super-secret-jwt-key-here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
MAIN_PORT=8000
MAIN_HOST=0.0.0.0
LOG_LEVEL=INFO
```

### Configuration Files
- `configs/config.yaml` - Main application configuration
- `configs/microservices.yaml` - Microservices configuration
- `configs/source.yaml` - Source configuration

## 📡 API Endpoints

### Health Check
- `GET /api/v1/health` - Application health status

### Authentication (TODO)
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/logout` - User logout
- `POST /api/v1/auth/refresh` - Refresh token
- `GET /api/v1/auth/me` - Get current user

### Document Processing (TODO)
- `POST /api/v1/documents/extract` - Extract document content
- `POST /api/v1/documents/upload` - Upload and extract file
- `POST /api/v1/documents/store` - Extract and store document

### RAG (TODO)
- `POST /api/v1/rag/query` - Process RAG query
- `POST /api/v1/rag/generate` - Generate response

### Vector Store (TODO)
- `POST /api/v1/vector/store` - Store documents
- `POST /api/v1/vector/search` - Search similar documents
- `GET /api/v1/vector/collections` - List collections

### Web Crawler (TODO)
- `POST /api/v1/crawler/crawl` - Crawl website
- `POST /api/v1/crawler/extract` - Extract single page

### Chat (TODO)
- `POST /api/v1/chat/message` - Send chat message
- `GET /api/v1/chat/conversations` - List conversations
- `GET /api/v1/chat/conversations/{id}/history` - Get conversation history

## 🧪 Testing

### Run Tests

**With Poetry:**
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app

# Run specific test file
poetry run pytest tests/test_main.py

# Run with verbose output
poetry run pytest -v
```

**Or with pip:**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_main.py

# Run with verbose output
pytest -v
```

### Test Structure
- `tests/test_main.py` - Main application tests
- `tests/test_api/test_v1/test_endpoints.py` - API endpoint tests
- `tests/test_api/test_v1/test_models.py` - Model validation tests

## 🗄️ Database

### Models
- `User` - User accounts and profiles
- `UserSession` - Active user sessions
- `UserToken` - JWT tokens and refresh tokens
- `PasswordReset` - Password reset functionality
- `ServiceHealth` - Service health tracking

### Database Operations
- Uses SQLAlchemy ORM
- Async database operations
- Repository pattern for data access
- Automatic table creation

## 🔐 Security

### Authentication
- JWT token-based authentication
- Password hashing with bcrypt
- Session management
- Token refresh mechanism

### Security Features
- CORS middleware
- Input validation with Pydantic
- Error handling
- Logging and monitoring

## 📊 Monitoring

### Health Checks
- Application health endpoint
- Database connectivity check
- Service status monitoring

### Logging
- Structured logging
- Log levels configuration
- Error tracking
- Performance monitoring

## 🚀 Deployment

### Production Considerations
1. **Environment Variables**: Set all required environment variables
2. **Database**: Use production database (PostgreSQL, MySQL)
3. **Security**: Configure proper CORS, HTTPS
4. **Monitoring**: Set up application monitoring
5. **Scaling**: Use load balancers and multiple instances

### Docker Deployment
```bash
# Build production image
docker build -t rag-chat-room-web:latest .

# Run with environment variables
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -e JWT_SECRET_KEY=your-secret \
  rag-chat-room-web:latest
```

## 🔄 Development

### Code Style
- Follow PEP 8
- Use type hints
- Write docstrings
- Use black for formatting
- Use isort for import sorting

### Adding New Features
1. Create service in `app/services/`
2. Add repository in `app/repositories/`
3. Define schemas in `app/schemas/`
4. Add API endpoints in `app/api/v1/endpoints.py`
5. Write tests in `tests/`

### Database Migrations
```bash
# Generate migration
alembic revision --autogenerate -m "Description"

# Apply migration
alembic upgrade head
```

## 📚 Documentation

- **API Documentation**: Available at `/docs` when running
- **Database Guide**: `app/core/DATABASE_GUIDE.md`
- **Microservices Guide**: `MICROSERVICES_README.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the API documentation at `/docs`

---

**RAG Chat Room Web** - Collaborative chat rooms powered by retrieval-augmented AI. 🚀