#!/usr/bin/env python3
"""
Database initialization script for RAG Chat Room Web.
Creates tables and initializes the database.
"""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add app to Python path
app_path = Path(__file__).parent.parent / "app"
if str(app_path) not in sys.path:
    sys.path.insert(0, str(app_path))

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.core.database_manager import initialize_service_database
from app.core.config import config
from app.core.logging import setup_logging

logger = setup_logging("database_init")


async def init_database():
    """Initialize database for all services."""
    try:
        logger.info("🚀 Starting database initialization...")
        
        # Get database URL
        db_url = config.get_database_url()
        logger.info(f"Database URL: {db_url}")
        
        # Initialize database for main service
        await initialize_service_database("main", db_url)
        logger.info("✅ Main service database initialized")
        
        # Initialize database for auth service
        await initialize_service_database("auth", db_url)
        logger.info("✅ Auth service database initialized")
        
        # Initialize database for document service
        await initialize_service_database("document", db_url)
        logger.info("✅ Document service database initialized")
        
        # Initialize database for rag service
        await initialize_service_database("rag", db_url)
        logger.info("✅ RAG service database initialized")
        
        # Initialize database for vector service
        await initialize_service_database("vector", db_url)
        logger.info("✅ Vector service database initialized")
        
        # Initialize database for web crawler service
        await initialize_service_database("web_crawler", db_url)
        logger.info("✅ Web crawler service database initialized")
        
        # Initialize database for chat service
        await initialize_service_database("chat", db_url)
        logger.info("✅ Chat service database initialized")
        
        logger.info("🎉 Database initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(init_database())
