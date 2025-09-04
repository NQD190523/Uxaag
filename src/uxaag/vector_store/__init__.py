"""
Vector Store Module for UXAAG

This module handles document embeddings, vector storage, and similarity search.
It's the foundation of the RAG (Retrieval-Augmented Generation) system.

Key Components:
- DocumentEmbedder: Converts text to vector embeddings
- VectorDatabase: Stores and queries document embeddings
- DocumentChunker: Splits documents into searchable chunks

Usage:
    from uxaag.vector_store import DocumentEmbedder, VectorDatabase, DocumentChunker
    
    # Create embedder
    embedder = DocumentEmbedder()
    
    # Create vector database
    vector_db = VectorDatabase()
    
    # Create chunker
    chunker = DocumentChunker()
"""

# Add src to Python path for imports
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import modules using absolute imports
from uxaag.vector_store.embedder import DocumentEmbedder
from uxaag.vector_store.vector_db import VectorDatabase
from uxaag.vector_store.chunker import DocumentChunker

__all__ = [
    'DocumentEmbedder',
    'VectorDatabase', 
    'DocumentChunker'
] 