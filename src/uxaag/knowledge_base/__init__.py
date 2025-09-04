"""
Knowledge Base Module for UXAAG

This module manages document storage, indexing, and retrieval for the RAG system.
It acts as the central hub for all document operations, coordinating between
the vector database and document processing components.

Key Components:
- DocumentStore: Manages document storage and metadata
- Retriever: Handles document retrieval based on queries
- Indexer: Processes and indexes new documents

The knowledge base is the bridge between:
1. Raw documents (from web crawling, file uploads, etc.)
2. Processed chunks (split and cleaned documents)
3. Vector embeddings (for similarity search)
4. Retrieval results (for RAG generation)

Usage:
    from uxaag.knowledge_base import DocumentStore, Retriever, Indexer
    
    # Create knowledge base components
    doc_store = DocumentStore()
    retriever = Retriever(vector_db, embedder)
    indexer = Indexer(chunker, embedder, vector_db)
"""

# Add src to Python path for imports
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import modules using absolute imports
from uxaag.knowledge_base.retriever import Retriever

__all__ = [
    'Retriever'
] 