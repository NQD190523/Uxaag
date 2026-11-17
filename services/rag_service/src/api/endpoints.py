"""RAG endpoints router (router layer): ``/api/v1/rag/...``.

NOTE: stub. ``POST /rag/query`` returns a fixed message and performs no retrieval; real
retrieval (embeddings + vector store) is not built yet.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query")
async def rag_query():
    """POST /rag/query (full path ``/api/v1/rag/query``) - placeholder for RAG queries.

    Auth: none checked here (the gateway protects the ``/rag`` prefix). Body: none.
    Returns: ``{"success": True, "message": ...}`` (static). Errors: none.
    """
    return {
        "success": True,
        "message": "RAG service is operational",
    }
