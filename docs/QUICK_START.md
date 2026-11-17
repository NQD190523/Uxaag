# RAG Chat Room Web microservices - Quick Start Guide

## 🚀 Start All Services (Choose One Method)

### Windows
```bash
start-services.bat
```

### Linux/Mac
```bash
chmod +x start-services.sh
./start-services.sh
```

### Docker
```bash
docker-compose -f docker-compose.microservices.yml up -d
```

---

## ✅ Verify Services are Running

```bash
curl http://localhost:8000/health
```

Expected response: Shows health status of all services

---

## 📝 Quick API Tests

### Register a User
```bash
curl -X POST http://localhost:8006/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "Test123456",
    "full_name": "Test User"
  }'
```

### Login
```bash
curl -X POST http://localhost:8006/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "Test123456"
  }'
```

### Verify Token
```bash
curl http://localhost:8006/api/v1/auth/verify/YOUR_TOKEN_HERE
```

---

## 📂 Project Structure

```
services/
├── api-gateway/        (Port 8000)
├── auth-service/       (Port 8006) ✅ Fully implemented
├── chat-service/       (Port 8005)
├── document-service/   (Port 8001)
├── rag-service/        (Port 8002)
├── vector-store-service/ (Port 8003)
└── web-crawler-service/ (Port 8004)
```

---

## 🔗 Service URLs

| Service | URL |
|---------|-----|
| API Gateway | http://localhost:8000 |
| Document Service | http://localhost:8001 |
| RAG Service | http://localhost:8002 |
| Vector Store Service | http://localhost:8003 |
| Web Crawler Service | http://localhost:8004 |
| Chat Service | http://localhost:8005 |
| Auth Service | http://localhost:8006 |

---

## 📋 Key Files

- **MICROSERVICES_IMPLEMENTATION.md** - Complete implementation guide
- **MICROSERVICES_SETUP.md** - Detailed setup instructions
- **API_SPECIFICATION.md** - API endpoint specifications
- **docker-compose.microservices.yml** - Docker Compose configuration
- **configs/microservices.yaml** - Service configuration

---

## 🔧 Individual Service Startup

Each service can be started independently:

```bash
# Auth Service
cd services/auth-service
python main.py

# Chat Service (in another terminal)
cd services/chat-service
python main.py

# etc...
```

---

## 📊 Database Locations

Each service has its own database in the `data/` folder:
- `data/rag_chat_room_web_auth.db` - Auth Service
- `data/rag_chat_room_web_chat.db` - Chat Service
- `data/rag_chat_room_web_document.db` - Document Service
- `data/rag_chat_room_web_rag.db` - RAG Service
- `data/rag_chat_room_web_vector.db` - Vector Service
- `data/rag_chat_room_web_crawler.db` - Web Crawler Service

---

## 🛑 Stop Services

### From Windows batch windows
Close each window individually, or use:
```bash
taskkill /F /IM python.exe
```

### From Docker
```bash
docker-compose -f docker-compose.microservices.yml down
```

### From Linux/Mac
```bash
pkill -f "python main.py"
```

---

## 🔍 View Logs

### Linux/Mac
```bash
tail -f logs/*.log
```

### Windows (from services directory)
Each service window shows logs in real-time

### Docker
```bash
docker-compose -f docker-compose.microservices.yml logs -f
```

---

## ✨ What's Included

✅ 7 independent microservices with own API endpoints
✅ Service-specific database connections
✅ Centralized API Gateway for routing
✅ Complete Auth Service implementation
✅ Health check endpoints for all services
✅ Docker support with Dockerfiles
✅ Docker Compose configuration
✅ Startup scripts for Windows/Linux
✅ Comprehensive documentation
✅ API specifications

---

## 📚 Next Steps

1. **Review Documentation**
   - Read `MICROSERVICES_IMPLEMENTATION.md`
   - Check `API_SPECIFICATION.md`

2. **Start Services**
   - Run `start-services.bat` (Windows) or `./start-services.sh` (Linux/Mac)

3. **Test Auth Service**
   - Use the API test commands above

4. **Implement Remaining Services**
   - Each service has skeleton structure ready
   - Add business logic to endpoints

5. **Deploy**
   - Use Docker Compose for orchestration
   - Or deploy individually to production

---

## ⚙️ Configuration

Create `.env` file in root directory:

```env
GITHUB_TOKEN=your_token
OPENAI_API_KEY=your_key
DATABASE_URL=sqlite:///./data/rag_chat_room_web.db
REDIS_URL=redis://localhost:6379
```

---

## 🆘 Troubleshooting

**Port already in use?**
```bash
# Find what's using the port
lsof -i :8000  # Linux/Mac

# Change port in service main.py
```

**Module not found?**
```bash
pip install -r requirements.txt
```

**Database locked?**
```bash
# Close all service processes and try again
pkill -f "python main.py"
```

---

## 📞 Support

Refer to detailed documentation:
- Full Implementation Guide: `MICROSERVICES_IMPLEMENTATION.md`
- Setup Instructions: `MICROSERVICES_SETUP.md`
- API Endpoints: `API_SPECIFICATION.md`

All original code logic is preserved - only structure was reorganized!
