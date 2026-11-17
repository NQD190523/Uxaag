#!/usr/bin/env python3
"""
RAG Chat Room Web API Client

Simple client for testing the RAG Chat Room Web API endpoints.
"""

import requests
import json
from typing import List, Dict, Any

class RagChatRoomWebAPIClient:
    """Client for interacting with the RAG Chat Room Web API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def chat(self, question: str, session_id: str = "default", use_rag: bool = True) -> Dict[str, Any]:
        """Send a chat message."""
        data = {
            "question": question,
            "session_id": session_id,
            "use_rag": use_rag
        }
        response = self.session.post(f"{self.base_url}/chat", json=data)
        response.raise_for_status()
        return response.json()
    
    def batch_chat(self, questions: List[str], session_id: str = "batch", use_rag: bool = True) -> Dict[str, Any]:
        """Send multiple chat messages."""
        data = {
            "questions": questions,
            "session_id": session_id,
            "use_rag": use_rag
        }
        response = self.session.post(f"{self.base_url}/chat/batch", json=data)
        response.raise_for_status()
        return response.json()
    
    def rag_query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system directly."""
        data = {"question": question}
        response = self.session.post(f"{self.base_url}/rag/query", json=data)
        response.raise_for_status()
        return response.json()
    
    def crawl_single(self, url: str, requirements: str = "Extract main content and key information") -> Dict[str, Any]:
        """Crawl a single URL."""
        data = {
            "url": url,
            "requirements": requirements
        }
        response = self.session.post(f"{self.base_url}/crawl/single", json=data)
        response.raise_for_status()
        return response.json()
    
    def crawl_multiple(self, urls: List[str], requirements: str = "Extract main content and key information") -> Dict[str, Any]:
        """Crawl multiple URLs."""
        data = {
            "urls": urls,
            "requirements": requirements
        }
        response = self.session.post(f"{self.base_url}/crawl/multiple", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_vector_db_status(self) -> Dict[str, Any]:
        """Get vector database status."""
        response = self.session.get(f"{self.base_url}/vector-db/status")
        response.raise_for_status()
        return response.json()
    
    def add_document(self, content: str, title: str = None, category: str = "general", metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a document to the vector database."""
        data = {
            "content": content,
            "title": title,
            "category": category,
            "metadata": metadata or {}
        }
        response = self.session.post(f"{self.base_url}/vector-db/add-document", json=data)
        response.raise_for_status()
        return response.json()
    
    def add_sample_documents(self) -> Dict[str, Any]:
        """Add sample documents to the vector database."""
        response = self.session.post(f"{self.base_url}/vector-db/add-samples")
        response.raise_for_status()
        return response.json()
    
    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.5) -> Dict[str, Any]:
        """Search documents in the vector database."""
        data = {
            "query": query,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold
        }
        response = self.session.post(f"{self.base_url}/search", json=data)
        response.raise_for_status()
        return response.json()


def test_api():
    """Test the API endpoints."""
    print("🧪 Testing RAG Chat Room Web API")
    print("=" * 50)
    
    client = RagChatRoomWebAPIClient()
    
    try:
        # Test health check
        print("1. Testing health check...")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Components: {health['components']}")
        
        # Test chat
        print("\n2. Testing chat...")
        chat_response = client.chat("How do I summarize this chat room?")
        print(f"   Success: {chat_response['success']}")
        print(f"   Answer: {chat_response['answer'][:100]}...")
        
        # Test RAG query
        print("\n3. Testing RAG query...")
        rag_response = client.rag_query("What are the key points from the uploaded documents?")
        print(f"   Success: {rag_response['success']}")
        print(f"   Answer: {rag_response['answer'][:100]}...")
        print(f"   Sources: {rag_response['sources']}")
        
        # Test vector DB status
        print("\n4. Testing vector DB status...")
        db_status = client.get_vector_db_status()
        print(f"   Total documents: {db_status['status']['total_documents']}")
        print(f"   Collection: {db_status['status']['collection_name']}")
        
        # Test search
        print("\n5. Testing search...")
        search_response = client.search("RAG chat room best practices")
        print(f"   Found {len(search_response['results'])} results")
        if search_response['results']:
            print(f"   First result: {search_response['results'][0]['content'][:100]}...")
        
        print("\n✅ All API tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ API test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_api()

