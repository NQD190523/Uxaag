"""
Document Embedder for UXAAG

This module handles converting text documents into vector embeddings.
Embeddings are numerical representations of text that capture semantic meaning,
allowing us to find similar documents through vector similarity search.

Key Concepts:
- Embeddings are high-dimensional vectors (typically 384-1536 dimensions)
- Similar documents have similar embedding vectors
- We use sentence-transformers for generating embeddings
- Embeddings enable semantic search (finding meaning, not just keywords)

Dependencies to add to requirements.txt:
- sentence-transformers
- torch
- numpy
"""

from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

class DocumentEmbedder:
    """
    Converts text documents into vector embeddings for similarity search.
    
    This is the core component that enables semantic search in your RAG system.
    Instead of keyword matching, embeddings allow finding documents that are
    semantically similar to a query, even if they don't share exact words.
    
    Example:
        embedder = DocumentEmbedder()
        embeddings = embedder.embed_documents(["UX design principles", "User interface guidelines"])
        query_embedding = embedder.embed_query("How to design better user experiences?")
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize the document embedder.
        
        Args:
            model_name: Name of the sentence transformer model to use
                       - "all-MiniLM-L6-v2": Fast, good quality (384 dimensions)
                       - "all-mpnet-base-v2": Better quality, slower (768 dimensions)
                       - "text-embedding-ada-002": OpenAI's model (1536 dimensions)
            device: Device to run the model on ("cpu", "cuda", etc.)
        """
        self.model_name = model_name
        self.device = device
        
        # TODO: Initialize the sentence transformer model
        self.model = SentenceTransformer(model_name, device=device)
        
        # Get embedding dimensions for the chosen model
        self.embedding_dimensions = self._get_embedding_dimensions()
        
        logging.info(f"Initialized DocumentEmbedder with model: {model_name}")
    
    def _get_embedding_dimensions(self) -> int:
        """
        Get the embedding dimensions for the chosen model.
        
        Returns:
            Number of dimensions in the embedding vectors
        """
        # Get dimensions from the actual model
        try:
            # Create a dummy embedding to get the dimensions
            dummy_embedding = self.model.encode(["test"], convert_to_numpy=True)
            return dummy_embedding.shape[1]
        except Exception as e:
            logging.warning(f"Could not determine embedding dimensions from model: {e}")
            # Fallback to common dimensions
            if "MiniLM" in self.model_name:
                return 384
            elif "mpnet" in self.model_name:
                return 768
            elif "ada-002" in self.model_name:
                return 1536
            else:
                return 384  # Default
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Convert a list of documents into embeddings.
        
        This method takes multiple documents and converts them to vectors.
        The embeddings can then be stored in a vector database for later retrieval.
        
        Args:
            documents: List of text documents to embed
            
        Returns:
            numpy array of shape (num_documents, embedding_dimensions)
            
        Example:
            documents = [
                "UX design focuses on user experience",
                "UI design is about visual interface",
                "Accessibility ensures inclusive design"
            ]
            embeddings = embedder.embed_documents(documents)
            # Returns: array([[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]])
        """
        if not documents:
            return np.array([])
        
        # Use the sentence transformer model to encode documents
        embeddings = self.model.encode(documents, convert_to_numpy=True)
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Convert a single query into an embedding.
        
        This method is used when a user asks a question. The query embedding
        is compared against document embeddings to find the most relevant documents.
        
        Args:
            query: The user's question or search query
            
        Returns:
            numpy array of shape (embedding_dimensions,)
            
        Example:
            query = "What are the best UX design practices?"
            query_embedding = embedder.embed_query(query)
            # Returns: array([0.1, 0.2, 0.3, ...])
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Use the sentence transformer model to encode the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        return query_embedding[0]  # Return the first (and only) embedding
    
    def compute_similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity scores between a query and multiple documents.
        
        This method calculates how similar each document is to the query.
        Higher scores indicate more relevant documents.
        
        Args:
            query_embedding: Embedding of the user's query
            document_embeddings: Embeddings of documents to compare against
            
        Returns:
            numpy array of similarity scores (higher = more similar)
            
        Example:
            similarities = embedder.compute_similarity(query_emb, doc_embs)
            # Returns: array([0.85, 0.72, 0.91, 0.63])
        """
        # TODO: Implement similarity computation
        # Common methods:
        # 1. Cosine similarity: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        # 2. Euclidean distance: np.linalg.norm(a - b)
        # 3. Dot product: np.dot(a, b)
        
        # For cosine similarity:
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(document_embeddings, axis=1)
        similarities = np.dot(document_embeddings, query_embedding) / (doc_norms * query_norm)
        
        # Placeholder implementation
        num_docs = document_embeddings.shape[0]
        return np.random.rand(num_docs) 