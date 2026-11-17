#!/bin/bash

# Startup script for RAG Chat Room Web microservices
# This script starts all microservices in the background

echo "🚀 Starting RAG Chat Room Web microservices..."

# This script lives in uxaag/scripts/; run from the uxaag/ folder.
cd "$(dirname "$0")/.." || exit 1

# Function to start a service
start_service() {
    local service_name=$1
    local port=$2
    local dir=$3
    
    echo "Starting $service_name on port $port..."
    cd "$dir" || exit
    python main.py > "../../logs/${service_name}.log" 2>&1 &
    cd - > /dev/null
    echo "✅ $service_name started (PID: $!)"
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Start all services
start_service "auth-service" "8006" "services/auth_service"
start_service "chat-service" "8005" "services/chat_service"
start_service "document-service" "8001" "services/document_service"
start_service "rag-service" "8002" "services/rag_service"
start_service "vector-store-service" "8003" "services/vector_store_service"
start_service "web-crawler-service" "8004" "services/web_crawler_service"
start_service "api-gateway" "8000" "services/api_gateway"

echo ""
echo "🌐 All services started!"
echo ""
echo "Service URLs:"
echo "  API Gateway:           http://localhost:8000"
echo "  Document Service:      http://localhost:8001"
echo "  RAG Service:           http://localhost:8002"
echo "  Vector Store Service:  http://localhost:8003"
echo "  Web Crawler Service:   http://localhost:8004"
echo "  Chat Service:          http://localhost:8005"
echo "  Auth Service:          http://localhost:8006"
echo ""
echo "View logs:"
echo "  tail -f logs/*.log"
echo ""
echo "To stop all services: pkill -f 'python main.py'"
