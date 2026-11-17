# Database Architecture Guide for Microservices

This guide explains how to organize database connections cleanly and follow standard patterns in the RAG Chat Room Web microservices architecture.

## 🏗️ **Database Architecture Overview**

### **Pattern: Database per Service**
Each microservice has its own database instance, ensuring:
- **Data Isolation**: Services don't interfere with each other's data
- **Independent Scaling**: Each service can scale its database independently
- **Technology Flexibility**: Different services can use different database technologies
- **Fault Isolation**: Database issues in one service don't affect others

### **Database Organization**
```
data/
├── rag_chat_room_web_auth.db          # Authentication service database
├── rag_chat_room_web_document.db      # Document service database
├── rag_chat_room_web_rag.db           # RAG service database
├── rag_chat_room_web_vector.db        # Vector store service database
├── rag_chat_room_web_crawler.db       # Web crawler service database
└── rag_chat_room_web_chat.db          # Chat service database
```

## 🔧 **Database Components**

### **1. Database Models (`database_models.py`)**
- **Centralized Models**: All database models in one place
- **Service-Specific Models**: Each service can have its own models
- **Shared Models**: Common models used across services
- **Relationships**: Proper foreign key relationships

### **2. Database Manager (`database_manager.py`)**
- **Service-Specific Connections**: Each service gets its own database manager
- **Async Support**: Full async/await support for better performance
- **Connection Pooling**: Efficient connection management
- **Health Monitoring**: Database health checks
- **Auto Table Creation**: Automatic table creation on startup

### **3. Repository Pattern (`repositories.py`)**
- **Data Access Layer**: Clean separation between business logic and data access
- **Async Operations**: All database operations are async
- **Error Handling**: Proper error handling and logging
- **Transaction Management**: Automatic transaction handling

## 🚀 **Usage in Services**

### **1. Initialize Database in Service**

```python
from shared.database_manager import get_database_manager

class MyService:
    def __init__(self):
        self.db_manager = get_database_manager("my-service")
    
    async def initialize(self):
        # Initialize database connection
        self.db_manager.initialize()
        await self.db_manager.create_tables()
```

### **2. Use Repository Pattern**

```python
from shared.repositories import UserRepository

async def create_user(user_data):
    async with self.db_manager.get_async_session() as session:
        user_repo = UserRepository(session)
        user = await user_repo.create_user(
            username=user_data.username,
            email=user_data.email,
            hashed_password=user_data.hashed_password
        )
        return user
```

### **3. Handle Transactions**

```python
async def complex_operation():
    async with self.db_manager.get_async_session() as session:
        try:
            # Multiple operations in one transaction
            user_repo = UserRepository(session)
            session_repo = UserSessionRepository(session)
            
            user = await user_repo.create_user(...)
            session = await session_repo.create_session(...)
            
            # Transaction is automatically committed
            return {"user": user, "session": session}
        except Exception:
            # Transaction is automatically rolled back
            raise
```

## 📊 **Database Models**

### **User Management (Auth Service)**
- `User`: User accounts and profiles
- `UserSession`: Active user sessions
- `UserToken`: JWT tokens and refresh tokens
- `PasswordReset`: Password reset tokens

### **Service Health (All Services)**
- `ServiceHealth`: Service health tracking

### **Adding New Models**

```python
# In database_models.py
class MyNewModel(Base):
    __tablename__ = "my_new_table"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
```

## 🔄 **Database Migrations**

### **Automatic Table Creation**
Tables are automatically created when services start up:

```python
# In service initialization
await self.db_manager.create_tables()
```

### **Manual Migrations**
For complex schema changes, use Alembic:

```bash
# Generate migration
alembic revision --autogenerate -m "Add new column"

# Apply migration
alembic upgrade head
```

## 🛡️ **Security Best Practices**

### **1. Connection Security**
- Use connection pooling
- Implement connection timeouts
- Use environment variables for credentials
- Enable SSL in production

### **2. Data Protection**
- Hash sensitive data (passwords)
- Use proper data types
- Implement data validation
- Regular security audits

### **3. Access Control**
- Service-specific database access
- Principle of least privilege
- Regular access reviews

## 📈 **Performance Optimization**

### **1. Connection Pooling**
```python
# Configure connection pool
self.engine = create_async_engine(
    database_url,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=300
)
```

### **2. Query Optimization**
- Use proper indexes
- Implement query caching
- Use async operations
- Monitor query performance

### **3. Database Monitoring**
```python
# Health check
health_status = await self.db_manager.health_check()
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Database URL
DATABASE_URL=sqlite:///./data/rag_chat_room_web.db

# Service-specific database
AUTH_DATABASE_URL=sqlite:///./data/rag_chat_room_web_auth.db
DOCUMENT_DATABASE_URL=sqlite:///./data/rag_chat_room_web_document.db
```

### **Service Configuration**
```yaml
# configs/microservices.yaml
database:
  url: "sqlite:///./data/rag_chat_room_web.db"
  echo: false
  pool_size: 20
  max_overflow: 30
```

## 🧪 **Testing**

### **Test Database Setup**
```python
# Use in-memory database for tests
test_db_url = "sqlite+aiosqlite:///:memory:"
db_manager = DatabaseManager("test-service")
db_manager.initialize(test_db_url)
```

### **Database Fixtures**
```python
@pytest.fixture
async def test_db():
    db_manager = get_database_manager("test")
    db_manager.initialize("sqlite+aiosqlite:///:memory:")
    await db_manager.create_tables()
    yield db_manager
    await db_manager.close()
```

## 🚨 **Error Handling**

### **Database Connection Errors**
```python
try:
    async with self.db_manager.get_async_session() as session:
        # Database operations
        pass
except Exception as e:
    logger.error(f"Database operation failed: {str(e)}")
    # Handle error appropriately
```

### **Transaction Rollback**
```python
async with self.db_manager.get_async_session() as session:
    try:
        # Operations
        pass
    except Exception:
        # Automatic rollback
        raise
```

## 📚 **Best Practices Summary**

1. **Use Repository Pattern**: Clean separation of data access logic
2. **Async Operations**: Use async/await for all database operations
3. **Connection Management**: Use context managers for automatic cleanup
4. **Error Handling**: Proper error handling and logging
5. **Health Monitoring**: Regular database health checks
6. **Security**: Hash sensitive data, use proper authentication
7. **Performance**: Use connection pooling and query optimization
8. **Testing**: Use test databases for unit tests
9. **Documentation**: Document all database models and operations
10. **Monitoring**: Monitor database performance and health

## 🔄 **Migration from Monolith**

When migrating from a monolithic database:

1. **Identify Data Ownership**: Determine which service owns which data
2. **Create Service Databases**: Set up separate databases for each service
3. **Data Migration**: Migrate data to appropriate service databases
4. **Update Services**: Update services to use new database structure
5. **Test Thoroughly**: Test all database operations
6. **Monitor**: Monitor database performance and health

This architecture provides a clean, scalable, and maintainable database structure for your microservices! 🚀
