"""
Vector Database for UXAAG

This module provides a vector database interface for storing and querying document embeddings.
A vector database is essential for RAG systems as it enables fast similarity search
across large collections of documents.

Key Concepts:
- Vector storage: Storing high-dimensional embeddings efficiently
- Similarity search: Finding documents similar to a query vector
- Indexing: Optimizing search performance with data structures like HNSW, IVF
- Persistence: Saving embeddings to disk for reuse

Vector Database Options:
1. ChromaDB: Simple, in-memory or persistent
2. FAISS: Fast, by Facebook/Meta
3. Pinecone: Cloud-based, managed service
4. Qdrant: Open-source, feature-rich
5. Weaviate: Graph-based vector database

For this project, we'll start with ChromaDB as it's simple to set up.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from pathlib import Path

# TODO: Add these imports when you implement the actual vector database
import chromadb
from chromadb.config import Settings

class VectorDatabase:
    """
    Interface for storing and querying document embeddings.
    
    This class abstracts the vector database operations, making it easy to
    switch between different vector database implementations (ChromaDB, FAISS, etc.).
    
    The vector database stores:
    1. Document embeddings (vectors)
    2. Document metadata (source, title, etc.)
    3. Document IDs for retrieval
    
    Example:
        vector_db = VectorDatabase("data/vector_store")
        vector_db.add_documents(chunks, embeddings)
        results = vector_db.search(query_embedding, top_k=5)
    """
    
    def __init__(
        self,
        persist_directory: str = "data/vector_store",
        collection_name: str = "uxaag_documents",
        embedding_dimensions: int = 384
    ):
        """
        Initialize the vector database.
        
        Args:
            persist_directory: Directory to store the vector database
            collection_name: Name of the document collection
            embedding_dimensions: Number of dimensions in the embeddings
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_dimensions = embedding_dimensions
        
        # Create directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # TODO: Initialize the actual vector database
        # For ChromaDB:
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logging.info(f"Initialized VectorDatabase: {self.persist_directory}")
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray,
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add documents and their embeddings to the vector database.
        
        This method stores both the document content and its vector embedding
        for later retrieval. The embeddings should be generated using the
        DocumentEmbedder class.
        
        Args:
            documents: List of documents with content and metadata
            embeddings: numpy array of document embeddings
            ids: Optional list of document IDs (auto-generated if not provided)
            
        Returns:
            True if successful, False otherwise
            
        Example:
            documents = [
                {"content": "UX design principles", "metadata": {"source": "nngroup.com"}},
                {"content": "User interface guidelines", "metadata": {"source": "uxmatters.com"}}
            ]
            embeddings = embedder.embed_documents([doc["content"] for doc in documents])
            vector_db.add_documents(documents, embeddings)
        """
        if not documents or len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have the same length")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # TODO: Implement document addition
        # For ChromaDB:
        contents = [doc["content"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        
        self.collection.add(
            documents=contents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        logging.info(f"Added {len(documents)} documents to vector database")
        return True
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        This method finds the most similar documents to a query embedding
        by computing similarity scores and returning the top-k results.
        
        Args:
            query_embedding: Vector representation of the search query
            top_k: Number of top results to return
            filter_metadata: Optional metadata filter (e.g., {"source": "nngroup.com"})
            
        Returns:
            List of dictionaries containing document info and similarity scores
            
        Example:
            query_embedding = embedder.embed_query("What are UX design principles?")
            results = vector_db.search(query_embedding, top_k=3)
            # Returns: [{"content": "...", "metadata": {...}, "score": 0.85}, ...]
        """
        if query_embedding.shape[0] != self.embedding_dimensions:
            raise ValueError(f"Query embedding must have {self.embedding_dimensions} dimensions")
        
        # Use ChromaDB for similarity search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_metadata
        )
        
        # Format results to match what main.py expects
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "page_content": results["documents"][0][i],  # This is what main.py expects
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {},
                    "score": results["distances"][0][i] if results["distances"] and results["distances"][0] else 0.0,
                    "id": results["ids"][0][i] if results["ids"] and results["ids"][0] else f"doc_{i}"
                })
        
        return formatted_results
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Document information or None if not found
        """
        # TODO: Implement document retrieval by ID
        # For ChromaDB:
        results = self.collection.get(ids=[document_id])
        if results["documents"]:
            return {
                "content": results["documents"][0],
                "metadata": results["metadatas"][0],
                "id": results["ids"][0]
            }
        return None
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents from the vector database.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        # TODO: Implement document deletion
        # For ChromaDB:
        self.collection.delete(ids=document_ids)
        
        logging.info(f"Deleted {len(document_ids)} documents from vector database")
        return True
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the document collection.
        
        Returns:
            Dictionary with collection statistics
        """
        # TODO: Implement collection statistics
        # For ChromaDB:
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "embedding_dimensions": self.embedding_dimensions
        }
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        # TODO: Implement collection clearing
        # For ChromaDB:
        self.collection.delete(where={})
        
        logging.info("Cleared all documents from vector database")
        return True 