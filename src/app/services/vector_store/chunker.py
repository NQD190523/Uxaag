"""
Document Chunker for UXAAG

This module handles splitting large documents into smaller, searchable chunks.
Chunking is crucial for RAG systems because:
1. LLMs have token limits for context
2. Smaller chunks allow more precise retrieval
3. Different parts of a document may be relevant to different queries

Key Concepts:
- Chunk size: Number of characters/tokens per chunk
- Chunk overlap: Overlapping text between chunks to maintain context
- Chunking strategy: How to split documents (by sentences, paragraphs, etc.)
- Metadata preservation: Keeping track of chunk source and position

Example:
    A 10,000 word article about UX design might be split into 20 chunks
    of 500 words each, with 50 words of overlap between chunks.
"""

from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass
import logging

@dataclass
class DocumentChunk:
    """
    Represents a single chunk of a document.
    
    Attributes:
        content: The text content of the chunk
        metadata: Information about the chunk (source, position, etc.)
        chunk_id: Unique identifier for the chunk
    """
    content: str
    metadata: Dict[str, Any]
    chunk_id: str

class DocumentChunker:
    """
    Splits documents into smaller chunks for better retrieval.
    
    This class implements different chunking strategies to break down
    large documents into manageable pieces that can be:
    1. Embedded into vectors
    2. Stored in a vector database
    3. Retrieved based on similarity to user queries
    
    Example:
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_document("A very long UX design article...")
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        chunk_strategy: str = "sentence"
    ):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Target size of each chunk (in characters)
            chunk_overlap: Number of characters to overlap between chunks
            chunk_strategy: How to split documents ("sentence", "paragraph", "fixed")
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        
        # Validate parameters
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        logging.info(f"Initialized DocumentChunker: size={chunk_size}, overlap={chunk_overlap}, strategy={chunk_strategy}")
    
    def chunk_document(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Split a document into chunks.
        
        This is the main method that takes a document and splits it into
        smaller pieces. The chunking strategy determines how the splitting
        is done.
        
        Args:
            content: The document text to chunk
            metadata: Optional metadata to attach to all chunks
            
        Returns:
            List of DocumentChunk objects
            
        Example:
            content = "UX design is about creating user-centered experiences..."
            chunks = chunker.chunk_document(content, {"source": "nngroup.com"})
        """
        if not content.strip():
            return []
        
        metadata = metadata or {}
        
        # Choose chunking strategy
        if self.chunk_strategy == "sentence":
            return self._chunk_by_sentences(content, metadata)
        elif self.chunk_strategy == "paragraph":
            return self._chunk_by_paragraphs(content, metadata)
        elif self.chunk_strategy == "fixed":
            return self._chunk_by_fixed_size(content, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.chunk_strategy}")
    
    def _chunk_by_sentences(self, content: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Split document by sentences, respecting chunk size limits.
        
        This strategy tries to keep sentences together, which is often
        better for maintaining context and meaning.
        
        Args:
            content: Document content to chunk
            metadata: Metadata for the chunks
            
        Returns:
            List of chunks split by sentences
        """
        # TODO: Implement sentence-based chunking
        # 1. Split content into sentences using regex or NLP
        # 2. Group sentences into chunks that fit within chunk_size
        # 3. Add overlap between chunks
        # 4. Create DocumentChunk objects with metadata
        
        # Example implementation:
        sentences = re.split(r'[.!?]+', content)
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(DocumentChunk(
                        content=current_chunk.strip(),
                        metadata={**metadata, "chunk_index": chunk_id},
                        chunk_id=f"{metadata.get('source', 'doc')}_{chunk_id}"
                    ))
                    chunk_id += 1
                current_chunk = sentence + ". "
        
        # Placeholder implementation
        return [DocumentChunk(
            content=content[:self.chunk_size],
            metadata={**metadata, "chunk_index": 0},
            chunk_id=f"{metadata.get('source', 'doc')}_0"
        )]
    
    def _chunk_by_paragraphs(self, content: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Split document by paragraphs.
        
        This strategy is useful for documents with clear paragraph structure,
        like articles or blog posts.
        
        Args:
            content: Document content to chunk
            metadata: Metadata for the chunks
            
        Returns:
            List of chunks split by paragraphs
        """
        # TODO: Implement paragraph-based chunking
        # 1. Split content by double newlines or paragraph markers
        # 2. Group paragraphs into chunks
        # 3. Handle cases where paragraphs are too large or small
        
        # Placeholder implementation
        return [DocumentChunk(
            content=content[:self.chunk_size],
            metadata={**metadata, "chunk_index": 0},
            chunk_id=f"{metadata.get('source', 'doc')}_0"
        )]
    
    def _chunk_by_fixed_size(self, content: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Split document into fixed-size chunks.
        
        This is the simplest strategy but may break sentences or paragraphs
        in the middle. Use this when you need consistent chunk sizes.
        
        Args:
            content: Document content to chunk
            metadata: Metadata for the chunks
            
        Returns:
            List of fixed-size chunks
        """
        # TODO: Implement fixed-size chunking
        # 1. Split content into chunks of exactly chunk_size
        # 2. Add overlap between consecutive chunks
        # 3. Handle the last chunk (may be smaller)
        
        chunks = []
        chunk_id = 0
        start = 0
        
        while start < len(content):
            end = start + self.chunk_size
            chunk_content = content[start:end]
            
            chunks.append(DocumentChunk(
                content=chunk_content,
                metadata={**metadata, "chunk_index": chunk_id, "start": start, "end": end},
                chunk_id=f"{metadata.get('source', 'doc')}_{chunk_id}"
            ))
            
            chunk_id += 1
            start = end - self.chunk_overlap
        
        return chunks
    
    def chunk_multiple_documents(
        self, 
        documents: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """
        Chunk multiple documents at once.
        
        This is useful when you have a collection of documents to process.
        
        Args:
            documents: List of documents, each with 'content' and optional 'metadata'
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            doc_chunks = self.chunk_document(content, metadata)
            all_chunks.extend(doc_chunks)
        
        return all_chunks 