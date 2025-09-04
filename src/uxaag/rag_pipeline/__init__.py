"""
RAG Pipeline Module for UXAAG

This module orchestrates the complete RAG (Retrieval-Augmented Generation) workflow.
It combines retrieval, context building, and generation to create a complete
RAG system that can answer questions based on your knowledge base.

Key Components:
- RAGEngine: Main orchestrator that coordinates the entire RAG process
- PromptBuilder: Constructs prompts with retrieved context
- ResponseGenerator: Generates final responses using the LLM

The RAG Pipeline Flow:
1. User Query → Query Processing
2. Query → Document Retrieval
3. Retrieved Documents → Context Building
4. Context + Query → Prompt Construction
5. Prompt → LLM Generation
6. LLM Output → Response Formatting

This is where everything comes together to create a working RAG system.

Usage:
    from uxaag.rag_pipeline import RAGEngine, PromptBuilder, ResponseGenerator
    
    # Create RAG engine
    rag_engine = RAGEngine(retriever, llm, prompt_builder)
    
    # Generate RAG response
    response = rag_engine.generate_response("What are UX design principles?")
"""

# Add src to Python path for imports
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import modules using absolute imports
from uxaag.rag_pipeline.rag_engine import RAGEngine
from uxaag.rag_pipeline.prompt_builder import PromptBuilder
from uxaag.rag_pipeline.response_generator import ResponseGenerator

__all__ = [
    'RAGEngine',
    'PromptBuilder',
    'ResponseGenerator'
] 