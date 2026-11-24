"""Vector store HTTP endpoints (router layer), prefix ``/vectors``.

NOTE: not wired up. ``main.py`` never mounts this router; the manager lives in
``services/vector_manager.py``. This stub service has no ``pyproject.toml`` yet, so the
``shared`` lib must be on ``PYTHONPATH`` to import it. Intended full path would be
``/api/v1/vectors/...``.

Auth: none implemented (no ``x-user-id`` handling). Errors: most handlers wrap every
exception, including their own 400/404 ``HTTPException``, into a 500.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

from shared.logger import get_logger
from services.vector_manager import VectorStoreManager

logger = get_logger("vector-store-service")

# Create router
router = APIRouter(prefix="/vectors", tags=["vectors"])

# Global vector store instance
# Lazily created process-wide manager; in-memory, so data is lost on restart and not shared across workers.
vector_store: Optional[VectorStoreManager] = None


async def get_vector_store() -> VectorStoreManager:
    """Get or create vector store instance."""
    global vector_store
    if vector_store is None:
        vector_store = VectorStoreManager()
        await vector_store.initialize()
    return vector_store


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = {}


class StoreDocumentsRequest(BaseModel):
    """Request model for storing documents."""
    documents: List[DocumentMetadata] = Field(..., description="Documents to store")
    collection: Optional[str] = Field("default", description="Collection name")


class StoreDocumentsResponse(BaseModel):
    """Response model for storing documents."""
    success: bool
    document_ids: List[str]
    collection: str
    count: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SearchDocumentsRequest(BaseModel):
    """Request model for searching documents."""
    query: str = Field(..., min_length=1, description="Search query")
    collection: Optional[str] = Field("default", description="Collection to search")
    similarity_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    limit: Optional[int] = Field(10, ge=1, le=100)


class SearchResult(BaseModel):
    """Search result model."""
    id: str
    content: str
    metadata: Dict[str, Any]
    similarity: float


class SearchDocumentsResponse(BaseModel):
    """Response model for search results."""
    success: bool
    query: str
    results: List[SearchResult]
    count: int
    collection: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DeleteDocumentsRequest(BaseModel):
    """Request model for deleting documents."""
    doc_ids: List[str] = Field(..., description="Document IDs to delete")
    collection: Optional[str] = Field("default", description="Collection name")


# FIXME: ``store: VectorStoreManager = None`` is meant as an injected dependency but FastAPI
# will treat it as a request field; the same pattern repeats in the handlers below.
@router.post("/store", response_model=StoreDocumentsResponse)
async def store_documents(
    request: StoreDocumentsRequest,
    store: VectorStoreManager = None
):
    """POST /vectors/store - store documents (embeddings are mocked).

    Auth: none. Body: ``StoreDocumentsRequest`` (documents + collection, default ``default``).
    Returns: ``StoreDocumentsResponse`` with generated document ids (client-supplied ``id`` is ignored).
    Errors: 400 for an empty list (but see module note: surfaces as 500), 500 otherwise.
    """
    try:
        if not store:
            store = await get_vector_store()
        
        if not request.documents or len(request.documents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one document is required"
            )
        
        # Prepare documents
        docs_to_store = [
            {
                "content": doc.content,
                "metadata": doc.metadata or {}
            }
            for doc in request.documents
        ]
        
        # Store documents
        doc_ids = await store.store_documents(
            documents=docs_to_store,
            collection=request.collection or "default"
        )
        
        return StoreDocumentsResponse(
            success=True,
            document_ids=doc_ids,
            collection=request.collection or "default",
            count=len(doc_ids)
        )
    
    except Exception as e:
        logger.exception("Store documents failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store documents: {str(e)}"
        )


@router.post("/search", response_model=SearchDocumentsResponse)
async def search_documents(
    request: SearchDocumentsRequest,
    store: VectorStoreManager = None
):
    """POST /vectors/search - similarity search over a collection (mock embeddings).

    Auth: none. Body: ``SearchDocumentsRequest`` (query, collection, threshold, limit).
    Returns: ``SearchDocumentsResponse`` sorted by similarity.
    Errors: 400 for an empty query (surfaces as 500), 500 otherwise.
    """
    try:
        if not store:
            store = await get_vector_store()
        
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        # Search documents
        results = await store.search_documents(
            query=request.query,
            collection=request.collection or "default",
            similarity_threshold=request.similarity_threshold or 0.5,
            limit=request.limit or 10
        )
        
        search_results = [
            SearchResult(
                id=result["id"],
                content=result["content"],
                metadata=result["metadata"],
                similarity=result["similarity"]
            )
            for result in results
        ]
        
        return SearchDocumentsResponse(
            success=True,
            query=request.query,
            results=search_results,
            count=len(search_results),
            collection=request.collection or "default"
        )
    
    except Exception as e:
        logger.exception("Search documents failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document search failed: {str(e)}"
        )


@router.post("/delete")
async def delete_documents(
    request: DeleteDocumentsRequest,
    store: VectorStoreManager = None
):
    """POST /vectors/delete - delete documents by id (POST rather than DELETE, since it has a body).

    Auth: none. Body: ``DeleteDocumentsRequest``. Returns: ``{"success", "message", "deleted_count", "collection"}``.
    Errors: 400 for an empty id list (surfaces as 500), 500 otherwise.
    """
    try:
        if not store:
            store = await get_vector_store()
        
        if not request.doc_ids or len(request.doc_ids) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one document ID is required"
            )
        
        # Delete documents
        deleted_count = await store.delete_documents(
            doc_ids=request.doc_ids,
            collection=request.collection or "default"
        )
        
        return {
            "success": True,
            "message": f"Deleted {deleted_count} documents",
            "deleted_count": deleted_count,
            "collection": request.collection or "default"
        }
    
    except Exception as e:
        logger.exception("Delete documents failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete documents: {str(e)}"
        )


@router.get("/document/{doc_id}")
async def get_document(
    doc_id: str,
    collection: str = "default",
    store: VectorStoreManager = None
):
    """GET /vectors/document/{doc_id}?collection= - fetch one document (without its embedding).

    Auth: none. Returns: ``{"success", "document", "collection"}``.
    Errors: 404 if not found, 500 otherwise.
    """
    try:
        if not store:
            store = await get_vector_store()
        
        document = await store.get_document(doc_id=doc_id, collection=collection)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {doc_id} not found in collection {collection}"
            )
        
        return {
            "success": True,
            "document": document,
            "collection": collection
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Get document failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}"
        )


@router.get("/list")
async def list_documents(
    collection: str = "default",
    limit: int = 100,
    offset: int = 0,
    store: VectorStoreManager = None
):
    """GET /vectors/list?collection=&limit=&offset= - paginated listing (content truncated to 200 chars).

    Auth: none. Returns: ``{"success", "documents", "total", "limit", "offset", "collection"}``.
    Errors: 500 on failure.
    """
    try:
        if not store:
            store = await get_vector_store()
        
        result = await store.list_documents(
            collection=collection,
            limit=limit,
            offset=offset
        )
        
        return {
            "success": True,
            "documents": result.get("documents", []),
            "total": result.get("total", 0),
            "limit": limit,
            "offset": offset,
            "collection": collection
        }
    
    except Exception as e:
        logger.exception("List documents failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.post("/clear-collection")
async def clear_collection(
    collection: str = "default",
    store: VectorStoreManager = None
):
    """POST /vectors/clear-collection?collection= - remove every document in a collection.

    Auth: none. Irreversible (in-memory data). Returns: ``{"success", "message", "collection"}``.
    Errors: 500 on failure.
    """
    try:
        if not store:
            store = await get_vector_store()
        
        await store.clear_collection(collection=collection)
        
        return {
            "success": True,
            "message": f"Cleared collection '{collection}'",
            "collection": collection
        }
    
    except Exception as e:
        logger.exception("Clear collection failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear collection: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """GET /vectors/health - health probe that also lists the known collections.

    Auth: none. Returns: status, service, timestamp and collection names.
    Errors: 503 if the store cannot be created.
    """
    try:
        store = await get_vector_store()
        return {
            "status": "healthy",
            "service": "vector-store-service",
            "timestamp": datetime.utcnow().isoformat(),
            "collections": list(store.collections.keys())
        }
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store service is unhealthy"
        )
