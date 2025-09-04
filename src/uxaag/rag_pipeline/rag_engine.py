"""
RAG Engine for UXAAG

This module contains the main RAG (Retrieval-Augmented Generation) engine
that orchestrates the complete workflow from user query to final response.

The RAG Engine is the central coordinator that:
1. Takes user queries
2. Retrieves relevant documents
3. Builds context from retrieved documents
4. Generates responses using the LLM
5. Returns formatted answers

This is the heart of your RAG system - it's where all the components
come together to create a working AI assistant that can answer questions
based on your knowledge base.
"""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

# Add src to Python path for imports
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import modules using absolute imports
from uxaag.knowledge_base.retriever import Retriever
from uxaag.rag_pipeline.prompt_builder import PromptBuilder
from uxaag.rag_pipeline.response_generator import ResponseGenerator

@dataclass
class RAGResponse:
    """
    Represents a complete RAG response with all relevant information.
    
    Attributes:
        answer: The generated answer to the user's query
        retrieved_documents: List of documents used to generate the answer
        confidence_score: How confident the system is in the answer
        sources: List of sources used (URLs, document titles, etc.)
    """
    answer: str
    retrieved_documents: List[Dict[str, Any]]
    confidence_score: float
    sources: List[str]

class RAGEngine:
    """
    Main RAG engine that orchestrates the complete retrieval-augmented generation process.
    
    This class coordinates all the components of your RAG system:
    - Document retrieval based on user queries
    - Context building from retrieved documents
    - Prompt construction with relevant context
    - Response generation using the LLM
    
    The RAG process flow:
    1. User asks a question
    2. Engine retrieves relevant documents from knowledge base
    3. Engine builds context from retrieved documents
    4. Engine constructs a prompt with context and question
    5. LLM generates answer based on context
    6. Engine formats and returns the response
    
    Example:
        rag_engine = RAGEngine(retriever, llm, prompt_builder)
        response = rag_engine.generate_response("What are UX design principles?")
    """
    
    def __init__(
        self,
        retriever,  # Retriever
        llm,       # Language model
        prompt_builder,  # PromptBuilder
        max_context_length: int = 4000,
        top_k_retrieval: int = 5
    ):
        """
        Initialize the RAG engine.
        
        Args:
            retriever: Document retriever for finding relevant documents
            llm: Language model for generating responses
            prompt_builder: Component for building prompts with context
            max_context_length: Maximum length of context to include
            top_k_retrieval: Number of documents to retrieve
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.max_context_length = max_context_length
        self.top_k_retrieval = top_k_retrieval
        
        logging.info(f"Initialized RAGEngine: max_context={max_context_length}, top_k={top_k_retrieval}")
    
    def generate_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """
        Generate a RAG response for a user query.
        
        This is the main method that implements the complete RAG workflow:
        1. Retrieve relevant documents based on the query
        2. Build context from the retrieved documents
        3. Construct a prompt with the context and query
        4. Generate a response using the LLM
        5. Format and return the complete response
        
        Args:
            query: The user's question
            conversation_history: Optional conversation history for context
            filter_metadata: Optional metadata filter for document retrieval
            
        Returns:
            RAGResponse object with answer and metadata
            
        Example:
            query = "What are the best UX design practices?"
            response = rag_engine.generate_response(query)
            print(response.answer)  # The generated answer
            print(response.sources)  # Sources used
        """
        if not query.strip():
            return RAGResponse(
                answer="I couldn't process your query. Please provide a valid question.",
                retrieved_documents=[],
                confidence_score=0.0,
                sources=[]
            )
        
        try:
            # Step 1: Retrieve relevant documents
            logging.info(f"Retrieving documents for query: {query[:50]}...")
            retrieved_docs = self._retrieve_documents(query, filter_metadata)
            
            if not retrieved_docs:
                return RAGResponse(
                    answer="I couldn't find any relevant information to answer your question. Please try rephrasing your query or ask about a different topic.",
                    retrieved_documents=[],
                    confidence_score=0.0,
                    sources=[]
                )
            
            # Step 2: Build context from retrieved documents
            logging.info(f"Building context from {len(retrieved_docs)} documents")
            context = self._build_context(retrieved_docs)
            
            # Step 3: Construct prompt with context
            logging.info("Constructing prompt with context")
            prompt = self.prompt_builder.build_prompt(
                query=query,
                context=context,
                conversation_history=conversation_history
            )
            
            # Step 4: Generate response using LLM
            logging.info("Generating response with LLM")
            llm_response = self._generate_llm_response(prompt)
            
            # Step 5: Extract sources and calculate confidence
            sources = self._extract_sources(retrieved_docs)
            confidence = self._calculate_confidence(retrieved_docs, llm_response)
            
            # Step 6: Format and return response
            return RAGResponse(
                answer=llm_response,
                retrieved_documents=retrieved_docs,
                confidence_score=confidence,
                sources=sources
            )
            
        except Exception as e:
            logging.error(f"Error generating RAG response: {str(e)}")
            return RAGResponse(
                answer=f"I encountered an error while processing your question: {str(e)}. Please try again.",
                retrieved_documents=[],
                confidence_score=0.0,
                sources=[]
            )
    
    def _retrieve_documents(
        self,
        query: str,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for the query.
        
        Args:
            query: The user's question
            filter_metadata: Optional metadata filter
            
        Returns:
            List of retrieved documents with metadata
        """
        # TODO: Implement document retrieval
        # Use the retriever to get relevant documents
        retrieved_results = self.retriever.retrieve(
            query=query,
            top_k=self.top_k_retrieval,
            filter_metadata=filter_metadata
        )
        # # Convert to dictionary format
        documents = []
        for result in retrieved_results:
            documents.append({
                "content": result.content,
                "metadata": result.metadata,
                "score": result.score,
                "chunk_id": result.chunk_id
            })
        
        return documents
    
    def _build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved documents.
        
        This method combines the retrieved documents into a coherent
        context that can be used by the LLM to generate responses.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return ""
        
        # TODO: Implement context building
        # You can format the context in various ways:
        # 1. Simple concatenation with separators
        # 2. Structured format with document metadata
        # 3. Hierarchical organization by relevance
        
        # Example implementation:
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc["metadata"].get("source", "Unknown")
            content = doc["content"]
            context_parts.append(f"Document {i} (Source: {source}):\n{content}\n")
        
        context = "\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."
        
        return context
        
        # # Placeholder implementation
        # return "UX design focuses on creating user-centered experiences that are intuitive and accessible."
    
    def _generate_llm_response(self, prompt: str) -> str:
        """
        Generate response using the language model.
        
        Args:
            prompt: The constructed prompt with context
            
        Returns:
            Generated response from the LLM
        """
        # Use the response generator to get LLM response
        response = self.llm.generate_response(prompt)
        return response
    
    def _extract_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[str]:
        """
        Extract source information from retrieved documents.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            List of source identifiers (URLs, titles, etc.)
        """
        sources = []
        for doc in retrieved_docs:
            metadata = doc.get("metadata", {})
            source = metadata.get("source", metadata.get("title", "Unknown"))
            if source not in sources:
                sources.append(source)
        return sources
    
    def _calculate_confidence(
        self,
        retrieved_docs: List[Dict[str, Any]],
        llm_response: str
    ) -> float:
        """
        Calculate confidence score for the response.
        
        Args:
            retrieved_docs: Retrieved documents used
            llm_response: Generated response
            
        Returns:
            Confidence score between 0 and 1
        """
        # TODO: Implement confidence calculation
        # You can use various factors:
        # 1. Average similarity scores of retrieved documents
        # 2. Number of relevant documents found
        # 3. Response length and coherence
        # 4. Presence of specific keywords in response
        
        if not retrieved_docs:
            return 0.0
        
        # Simple confidence based on average document scores
        avg_score = sum(doc.get("score", 0) for doc in retrieved_docs) / len(retrieved_docs)
        return min(avg_score, 1.0)
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """
        Get statistics about RAG performance.
        
        Returns:
            Dictionary with RAG statistics
        """
        # TODO: Implement RAG statistics
        # You could track:
        # - Average response generation time
        # - Average number of documents retrieved
        # - Average confidence scores
        # - Most common query types
        
        return {
            "total_queries": 0,
            "average_response_time": 0.0,
            "average_documents_retrieved": 0,
            "average_confidence": 0.0
        } 