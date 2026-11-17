"""
Shared configuration management for microservices.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class Config:
    """Base configuration class for all microservices."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file) or {}
        except FileNotFoundError:
            return {}
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Support config['key'] syntax - loads from env vars first."""
        # Map common keys to env var names
        env_map = {
            'github_token': 'GITHUB_TOKEN',
            'openai_api_key': 'OPENAI_API_KEY',
            'database_url': 'DATABASE_URL',
            'redis_url': 'REDIS_URL',
            'llm_model': 'LLM_MODEL',
            'embedding_model': 'EMBEDDING_MODEL',
        }
        
        # Check env var first
        env_key = env_map.get(key, key.upper())
        env_value = os.getenv(env_key)
        if env_value:
            return env_value
        
        # Fall back to config file
        return self.get(key)
    
    def get_database_url(self) -> str:
        """Get database URL from environment or config."""
        return os.getenv("DATABASE_URL", self.get("database.url", "postgresql://username:password@localhost:5432/rag_chat_room_web"))
    
    def get_redis_url(self) -> str:
        """Get Redis URL from environment or config."""
        return os.getenv("REDIS_URL", self.get("redis.url", "redis://localhost:6379"))
    
    def get_vector_store_path(self) -> str:
        """Get vector store path."""
        return os.getenv("VECTOR_STORE_PATH", self.get("vector_store.path", "data/vector_store"))
    
    def get_embedding_model(self) -> str:
        """Get embedding model name."""
        return os.getenv("EMBEDDING_MODEL", self.get("embedding.model", "all-MiniLM-L6-v2"))
    
    def get_llm_model(self) -> str:
        """Get LLM model name."""
        return os.getenv("LLM_MODEL", self.get("llm.model", "gpt-3.5-turbo"))
    
    def get_openai_api_key(self) -> str:
        """Get OpenAI API key."""
        return os.getenv("OPENAI_API_KEY", self.get("openai.api_key", ""))
    
    def get_github_token(self) -> str:
        """Get GitHub token."""
        return os.getenv("GITHUB_TOKEN", self.get("github.token", ""))
    
    def get_service_port(self, service_name: str) -> int:
        """Get port for a specific service."""
        return int(os.getenv(f"{service_name.upper()}_PORT", self.get(f"services.{service_name}.port", 8000)))
    
    def get_service_host(self, service_name: str) -> str:
        """Get host for a specific service."""
        return os.getenv(f"{service_name.upper()}_HOST", self.get(f"services.{service_name}.host", "0.0.0.0"))
    
    def get_main_port(self) -> int:
        """Get main application port."""
        return int(os.getenv("MAIN_PORT", self.get("services.main.port", 8000)))
    
    def get_main_host(self) -> str:
        """Get main application host."""
        return os.getenv("MAIN_HOST", self.get("services.main.host", "0.0.0.0"))


# Global config instance
config = Config()
