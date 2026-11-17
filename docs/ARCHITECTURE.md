# RAG Chat Room Web microservices Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Applications                       │
│                  (Web, Mobile, Desktop, etc.)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    HTTP Requests (REST)
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API Gateway (Port 8000)                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • Request Routing                                       │   │
│  │ • Health Monitoring                                     │   │
│  │ • Load Balancing (Future)                              │   │
│  │ • Rate Limiting (Future)                               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─┬───┬──────┬─────────┬───────┬────────┬──────────────────────┬─┘
  │   │      │         │       │        │                      │
  │   │      │         │       │        │                      │
  ▼   ▼      ▼         ▼       ▼        ▼                      ▼
┌────────────────────────────────────────────────────────────────┐
│                   MICROSERVICES LAYER                           │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Document   │  │   RAG        │  │ Vector Store │         │
│  │   Service    │  │   Service    │  │   Service    │         │
│  │  (Port 8001) │  │  (Port 8002) │  │  (Port 8003) │         │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤         │
│  │ API Endpoints│  │ API Endpoints│  │ API Endpoints│         │
│  │ Database     │  │ Database     │  │ Database     │         │
│  │ Models       │  │ Models       │  │ Models       │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │     Web      │  │     Chat     │  │     Auth     │         │
│  │   Crawler    │  │    Service   │  │    Service   │         │
│  │   Service    │  │  (Port 8005) │  │  (Port 8006) │         │
│  │  (Port 8004) │  ├──────────────┤  ├──────────────┤         │
│  ├──────────────┤  │ API Endpoints│  │ API Endpoints│         │
│  │ API Endpoints│  │ Database     │  │ Database     │         │
│  │ Database     │  │ Models       │  │ Models       │         │
│  │ Models       │  │ Repositories │  │ Repositories │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
  │              │              │              │              │
  ▼              ▼              ▼              ▼              ▼
┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
│ Service 1 │ │ Service 2 │ │ Service 3 │ │ Service 4 │ │ Service 5 │
│ Database  │ │ Database  │ │ Database  │ │ Database  │ │ Database  │
│           │ │           │ │           │ │           │ │           │
│ rag_chat_    │ │ rag_chat_    │ │ rag_chat_    │ │ rag_chat_    │ │ rag_chat_    │
│ doc.db    │ │ rag.db    │ │ vector.db │ │ chat.db   │ │ auth.db   │
└───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘

                    ┌─────────────────┐
                    │     Redis       │
                    │  (Caching &     │
                    │  Sessions)      │
                    └─────────────────┘
```

## Service Communication Flow

```
┌──────────────┐
│ Client Calls │
│ API Gateway  │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────┐
│ API Gateway (Port 8000)                 │
│ • Routes request to appropriate service │
│ • Returns aggregated health status      │
└──────────┬──────────────────────────────┘
           │
           │ Forwards Request
           │
    ┌──────┴──────┬──────────┬────────────┬─────────────┬──────────┐
    │             │          │            │             │          │
    ▼             ▼          ▼            ▼             ▼          ▼
┌──────┐     ┌──────────┐ ┌──────┐  ┌────────┐  ┌─────────┐  ┌──────┐
│ Auth │     │   Chat   │ │ Doc  │  │  RAG   │  │ Vector  │  │ Web  │
│ Svc  │     │   Svc    │ │ Svc  │  │  Svc   │  │  Store  │  │Crawl │
└───┬──┘     └────┬─────┘ └───┬──┘  └───┬────┘  └────┬────┘  └───┬──┘
    │             │           │         │            │           │
    │ Read/Write  │           │         │            │           │
    │             │           │         │            │           │
    ▼             ▼           ▼         ▼            ▼           ▼
┌────────┐  ┌────────────┐ ┌──────┐ ┌──────┐  ┌──────────┐  ┌────────┐
│auth.db │  │  chat.db   │ │doc.db│ │rag.db│  │vector.db │  │crawler.db
└────────┘  └────────────┘ └──────┘ └──────┘  └──────────┘  └────────┘
```

## Service Responsibilities

```
┌─────────────────────────────────────────────────────────┐
│                   API Gateway                           │
│  Routes all client requests to appropriate services    │
│  Provides unified health monitoring                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   Auth Service                          │
│  • User Registration & Login                           │
│  • JWT Token Generation & Validation                   │
│  • Session Management                                  │
│  • Password Security (bcrypt + SHA256)                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   Chat Service                          │
│  • Chat Room Management                                │
│  • Message Storage & Retrieval                         │
│  • Real-time Messaging (WebSocket ready)              │
│  • Conversation History                                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                Document Service                         │
│  • Document Upload & Storage                           │
│  • File Format Processing                              │
│  • Text Extraction                                     │
│  • Document Metadata Management                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   RAG Service                           │
│  • Query Processing                                    │
│  • Document Retrieval                                  │
│  • LLM Integration                                     │
│  • Response Generation                                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              Vector Store Service                       │
│  • Vector Storage & Management                         │
│  • Embedding Generation                                │
│  • Similarity Search                                   │
│  • Vector Index Management                             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│             Web Crawler Service                         │
│  • URL Crawling & Scraping                             │
│  • Content Extraction                                  │
│  • Link Processing                                     │
│  • Data Normalization                                  │
└─────────────────────────────────────────────────────────┘
```

## Data Flow Example: User Registration

```
1. Client Request
   ┌──────────────────────────────┐
   │ POST /api/v1/auth/register   │
   │ {email, password, full_name} │
   └──────────────┬───────────────┘
                  │
                  ▼
2. API Gateway Routes Request
   ┌──────────────────────────────┐
   │ Routes to Auth Service:8006  │
   └──────────────┬───────────────┘
                  │
                  ▼
3. Auth Service Processes
   ┌─────────────────────────────────┐
   │ 1. Validate input              │
   │ 2. Check if email exists       │
   │ 3. Hash password               │
   │ 4. Create user in database     │
   │ 5. Return user profile         │
   └──────────────┬──────────────────┘
                  │
                  ▼
4. Database Writes
   ┌──────────────────────────────┐
   │ INSERT INTO users (         │
   │   email, hashed_password,    │
   │   full_name, created_at      │
   │ )                            │
   └──────────────┬───────────────┘
                  │
                  ▼
5. Response to Client
   ┌────────────────────────────────┐
   │ 201 Created                    │
   │ {                              │
   │   user_id,                     │
   │   email,                       │
   │   full_name,                   │
   │   is_active,                   │
   │   created_at                   │
   │ }                              │
   └────────────────────────────────┘
```

## Deployment Topology

### Development (Single Machine)
```
┌─────────────────────────────────────┐
│   Development Machine (localhost)   │
│                                     │
│ ┌─ Port 8000 ─────────────────┐   │
│ │ API Gateway                 │   │
│ └─────────────────────────────┘   │
│                                     │
│ ┌─ Port 8006 ──────────────────┐  │
│ │ Auth Service                 │  │
│ └─────────────────────────────┘   │
│                                     │
│ ┌─ Port 8005 ──────────────────┐  │
│ │ Chat Service                 │  │
│ └─────────────────────────────┘   │
│                                     │
│ ... other services ...              │
│                                     │
└─────────────────────────────────────┘
```

### Production (Docker)
```
┌────────────────────────────────────────────────┐
│           Docker Container Orchestration       │
│                (Docker Compose)                │
│                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Container │  │Container │  │Container │   │
│  │API GW    │  │Auth Svc  │  │Chat Svc  │   │
│  └──────────┘  └──────────┘  └──────────┘   │
│                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Container │  │Container │  │Container │   │
│  │Doc Svc   │  │RAG Svc   │  │Vector Svc│   │
│  └──────────┘  └──────────┘  └──────────┘   │
│                                                │
│  ┌──────────┐  ┌──────────┐                  │
│  │Container │  │Container │                  │
│  │Web Crawl │  │  Redis   │                  │
│  └──────────┘  └──────────┘                  │
│                                                │
│         Shared Volumes for Data               │
│          & Configuration Files                │
└────────────────────────────────────────────────┘
```

## Technology Stack

```
┌──────────────────────────────────────────────┐
│          RAG Chat Room Web microservices Stack           │
├──────────────────────────────────────────────┤
│                                              │
│  Language & Framework:                       │
│  • Python 3.11                               │
│  • FastAPI (async web framework)             │
│  • Pydantic (data validation)                │
│                                              │
│  Database:                                   │
│  • SQLAlchemy (ORM)                          │
│  • SQLite (development) / PostgreSQL (prod) │
│  • Async Session Management                 │
│                                              │
│  Authentication:                             │
│  • JWT (JSON Web Tokens)                     │
│  • bcrypt (password hashing)                 │
│  • JOSE (JWT handling)                       │
│                                              │
│  Caching & Sessions:                         │
│  • Redis                                     │
│                                              │
│  Containerization:                           │
│  • Docker                                    │
│  • Docker Compose                            │
│                                              │
│  API Documentation:                          │
│  • Swagger UI (auto-generated)               │
│  • ReDoc                                     │
│                                              │
│  Logging:                                    │
│  • Python logging module                     │
│  • Service-specific loggers                  │
│                                              │
└──────────────────────────────────────────────┘
```

## Scaling Strategy

```
┌────────────────────────────────────────────────┐
│        Independent Service Scaling             │
├────────────────────────────────────────────────┤
│                                                │
│  Service 1                                     │
│  ┌─────────────────┐                          │
│  │ Instance 1 (8006)                         │
│  ├─────────────────┤  ──┐                     │
│  │ Instance 2 (8006)   │ Load Balanced        │
│  ├─────────────────┤  ──┤                     │
│  │ Instance 3 (8006)   │                     │
│  └─────────────────┘  ──┘                     │
│                                                │
│  Service 2                                     │
│  ┌─────────────────┐                          │
│  │ Instance 1 (8005)                         │
│  ├─────────────────┤  ──┐                     │
│  │ Instance 2 (8005)   │ Load Balanced        │
│  └─────────────────┘  ──┘                     │
│                                                │
│  Each service can scale independently         │
│  based on load and requirements                │
│                                                │
└────────────────────────────────────────────────┘
```
