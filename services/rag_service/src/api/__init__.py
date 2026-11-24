"""Router package for rag_service: exports the ``/rag`` stub router and the ``/llm`` chat router."""

from api.endpoints import router as rag_router
from api.llm import router as llm_router

__all__ = ["rag_router", "llm_router"]
