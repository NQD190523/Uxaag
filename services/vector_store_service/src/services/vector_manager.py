"""In-memory vector store manager (service-style helper in the router package).

NOTE: not wired up and a mock. Embeddings are pseudo-random vectors (not semantic), data
lives in a Python dict and is lost on restart, and nothing imports this module because the
router that would use it is not mounted. A real implementation would call an embedding
model and persist to a vector database.
"""

import logging
from typing import List, Dict, Any, Optional
import uuid

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manage collections of documents with (mock) embeddings and cosine-similarity search.

    Data shape: ``collections[name] = {"name", "documents": {id: doc}, "metadata"}`` where each
    doc holds ``id``, ``content``, ``metadata``, ``embedding`` and ``stored_at``.
    """
    
    def __init__(self):
        """Initialize vector store manager."""
        self.collections: Dict[str, Dict[str, Any]] = {}
        self.default_collection = "default"
        self._ensure_collection(self.default_collection)
    
    async def initialize(self):
        """Async startup hook; currently only logs (a real backend would connect here)."""
        logger.info("Vector Store Manager initialized")
    
    def _ensure_collection(self, collection_name: str):
        """Ensure a collection exists."""
        if collection_name not in self.collections:
            self.collections[collection_name] = {
                "name": collection_name,
                "documents": {},
                "metadata": {}
            }
    
    async def store_documents(
        self,
        documents: List[Dict[str, Any]],
        collection: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Store documents with mock embeddings.

        Args:
            documents: ``[{"content": str, "metadata": dict}, ...]``.
            collection: Target collection (created if missing).
            metadata: Batch-level metadata; currently unused.

        Returns:
            The generated UUIDs, in input order.
        """
        try:
            self._ensure_collection(collection)
            doc_ids = []
            
            for doc in documents:
                doc_id = str(uuid.uuid4())
                
                # In production, generate embeddings here
                # For now, use mock embeddings
                # Embeddings are random, so similarity scores do not reflect meaning.
                mock_embedding = self._generate_mock_embedding(doc.get("content", ""))
                
                self.collections[collection]["documents"][doc_id] = {
                    "id": doc_id,
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "embedding": mock_embedding,
                    "stored_at": self._get_timestamp()
                }
                
                doc_ids.append(doc_id)
            
            logger.info(f"Stored {len(doc_ids)} documents in collection '{collection}'")
            return doc_ids
        
        except Exception as e:
            logger.error(f"Failed to store documents: {str(e)}")
            raise
    
    async def search_documents(
        self,
        query: str,
        collection: str = "default",
        similarity_threshold: float = 0.5,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Return documents whose similarity to the query is at least ``similarity_threshold``.

        Args:
            query: Search text.
            collection: Collection to search; unknown collections yield an empty list.
            similarity_threshold: Minimum cosine similarity.
            limit: Maximum number of results.

        Returns:
            ``[{"id", "content", "metadata", "similarity"}, ...]`` sorted by similarity, descending.
        """
        try:
            if collection not in self.collections:
                logger.warning(f"Collection '{collection}' not found")
                return []
            
            # Generate mock embedding for query
            query_embedding = self._generate_mock_embedding(query)
            
            # Brute-force scan over every document (O(n)); fine for a mock, not for a real index.
            # Calculate similarity for all documents
            results = []
            documents = self.collections[collection]["documents"]
            
            for doc_id, doc in documents.items():
                similarity = self._calculate_similarity(
                    query_embedding,
                    doc["embedding"]
                )
                
                if similarity >= similarity_threshold:
                    results.append({
                        "id": doc_id,
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "similarity": similarity
                    })
            
            # Sort by similarity and limit results
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            logger.info(f"Found {len(results)} documents matching query in collection '{collection}'")
            return results[:limit]
        
        except Exception as e:
            logger.error(f"Document search failed: {str(e)}")
            raise
    
    async def delete_documents(
        self,
        doc_ids: List[str],
        collection: str = "default"
    ) -> int:
        """
        Delete documents from vector store.
        
        Args:
            doc_ids: List of document IDs to delete
            collection: Collection name
        
        Returns:
            Number of documents deleted
        """
        try:
            if collection not in self.collections:
                return 0
            
            deleted_count = 0
            documents = self.collections[collection]["documents"]
            
            for doc_id in doc_ids:
                if doc_id in documents:
                    del documents[doc_id]
                    deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} documents from collection '{collection}'")
            return deleted_count
        
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            raise
    
    async def get_document(
        self,
        doc_id: str,
        collection: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID.
        
        Args:
            doc_id: Document ID
            collection: Collection name
        
        Returns:
            Document data or None if not found
        """
        try:
            if collection not in self.collections:
                return None
            
            documents = self.collections[collection]["documents"]
            doc = documents.get(doc_id)
            
            if doc:
                # Return without embedding for response
                return {
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "stored_at": doc["stored_at"]
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {str(e)}")
            raise
    
    async def list_documents(
        self,
        collection: str = "default",
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List all documents in a collection.
        
        Args:
            collection: Collection name
            limit: Maximum number of documents
            offset: Offset for pagination
        
        Returns:
            Dictionary with documents and metadata
        """
        try:
            if collection not in self.collections:
                return {"documents": [], "total": 0}
            
            documents = self.collections[collection]["documents"]
            all_docs = list(documents.values())
            
            total = len(all_docs)
            paginated_docs = all_docs[offset:offset+limit]
            
            # Return without embeddings
            result_docs = []
            for doc in paginated_docs:
                result_docs.append({
                    "id": doc["id"],
                    "content": doc["content"][:200],  # Preview
                    "metadata": doc["metadata"],
                    "stored_at": doc["stored_at"]
                })
            
            return {
                "documents": result_docs,
                "total": total,
                "limit": limit,
                "offset": offset,
                "collection": collection
            }
        
        except Exception as e:
            logger.error(f"Failed to list documents: {str(e)}")
            raise
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Return a 384-dimension pseudo-random vector derived from the text (mock, not semantic)."""
        # Simple hash-based mock embedding (not for production use)
        # NOTE: ``hash()`` of a str is salted per process (PYTHONHASHSEED), so the same text gets different vectors after a restart.
        hash_value = hash(text)
        import random
        # Seeding makes the vector deterministic within one process only.
        random.seed(hash_value % (2**31))
        return [random.random() for _ in range(384)]  # 384-dimensional embedding
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Cosine similarity of two vectors.

        Returns:
            A value in [-1, 1] (0.0 for empty or zero-norm inputs). Mock embeddings are all positive,
            so scores here land in [0, 1].
        """
        if not embedding1 or not embedding2:
            return 0.0
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _get_timestamp(self) -> str:
        """Return the current UTC time as an ISO-8601 string."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    async def clear_collection(self, collection: str = "default"):
        """Clear all documents from a collection."""
        try:
            if collection in self.collections:
                self.collections[collection]["documents"].clear()
                logger.info(f"Cleared collection '{collection}'")
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            raise
