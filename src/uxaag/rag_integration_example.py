"""
RAG Integration Example for UXAAG

This script demonstrates how to wire together all the RAG components:
1. Vector Database
2. Document Embedder
3. Document Chunker
4. Retriever
5. Prompt Builder
6. Response Generator
7. RAG Engine

Run this script to see how all components work together.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Import your RAG components
from uxaag.vector_store.vector_db import VectorDatabase
from uxaag.vector_store.embedder import DocumentEmbedder
from uxaag.vector_store.chunker import DocumentChunker
from uxaag.knowledge_base.retriever import Retriever
from uxaag.rag_pipeline.prompt_builder import PromptBuilder
from uxaag.rag_pipeline.response_generator import ResponseGenerator
from uxaag.rag_pipeline.rag_engine import RAGEngine

# Import your existing LLM setup
from uxaag.main import load_environment

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_rag_components():
    """
    Set up all RAG components and wire them together.
    
    Returns:
        RAGEngine instance ready to use
    """
    logger.info("Setting up RAG components...")
    
    # 1. Initialize Vector Database
    # Get the project root directory (two levels up from this file)
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    vector_store_path = project_root / "data" / "vector_store"
    
    vector_db = VectorDatabase(
        persist_directory=str(vector_store_path),
        collection_name="uxaag_documents"
    )
    logger.info("✓ Vector Database initialized")
    
    # 2. Initialize Document Embedder
    embedder = DocumentEmbedder(
        model_name="all-MiniLM-L6-v2",  # Fast, good quality
        device="cpu"  # Use "cuda" if you have GPU
    )
    logger.info("✓ Document Embedder initialized")
    
    # 3. Initialize Document Chunker
    chunker = DocumentChunker(
        chunk_size=500,
        chunk_overlap=50,
        chunk_strategy="sentence"
    )
    logger.info("✓ Document Chunker initialized")
    
    # 4. Initialize Retriever
    retriever = Retriever(
        vector_db=vector_db,
        embedder=embedder,
        top_k=5,
        similarity_threshold=0.5
    )
    logger.info("✓ Retriever initialized")
    
    # 5. Initialize Prompt Builder
    prompt_builder = PromptBuilder(
        system_instructions="You are UXAAG, a UX design AI assistant. Provide helpful, actionable UX advice based on the context provided.",
        max_history_length=3,
        context_format="structured"
    )
    logger.info("✓ Prompt Builder initialized")
    
    # 6. Initialize Response Generator with your existing LLM
    github_token = load_environment()
    from langchain_openai import AzureChatOpenAI
    
    llm = AzureChatOpenAI(
        api_key=github_token,
        api_version="2024-06-01",
        azure_endpoint="https://models.inference.ai.azure.com",
        model="gpt-4o",
        max_tokens=4000
    )
    
    response_generator = ResponseGenerator(
        llm=llm,
        max_retries=3,
        temperature=0.7,
        max_tokens=1000
    )
    logger.info("✓ Response Generator initialized")
    
    # 7. Initialize RAG Engine
    rag_engine = RAGEngine(
        retriever=retriever,
        llm=response_generator,  # Pass the response generator as the LLM
        prompt_builder=prompt_builder,
        max_context_length=4000,
        top_k_retrieval=5
    )
    logger.info("✓ RAG Engine initialized")
    
    return rag_engine, vector_db, embedder, chunker

def add_sample_documents(vector_db, embedder, chunker):
    """
    Add some sample UX design documents to the vector database.
    
    Args:
        vector_db: VectorDatabase instance
        embedder: DocumentEmbedder instance
        chunker: DocumentChunker instance
    """
    logger.info("Adding sample documents to the knowledge base...")
    
    # Sample UX design documents
    sample_docs = [
        {
            "content": """
            UX Design Principles: User Experience (UX) design focuses on creating products that provide meaningful and relevant experiences to users. The key principles include:
            1. User-Centered Design: Always design with the user in mind
            2. Consistency: Maintain consistent design patterns throughout the interface
            3. Accessibility: Ensure the design is usable by people with disabilities
            4. Usability: Make interfaces easy to learn and use
            5. Visual Hierarchy: Use visual elements to guide user attention
            """,
            "metadata": {
                "source": "ux_design_guide",
                "title": "UX Design Principles",
                "category": "fundamentals",
                "author": "UXAAG Team"
            }
        },
        {
            "content": """
            Color Theory in UX Design: Colors play a crucial role in user experience design. Understanding color psychology helps create effective interfaces:
            - Blue: Trust, stability, professionalism (use for corporate sites)
            - Green: Growth, harmony, nature (use for environmental products)
            - Red: Energy, urgency, passion (use sparingly for calls-to-action)
            - Yellow: Optimism, clarity, warmth (use for highlighting important information)
            - Purple: Luxury, creativity, mystery (use for creative industries)
            Always ensure sufficient color contrast for accessibility.
            """,
            "metadata": {
                "source": "color_psychology_guide",
                "title": "Color Theory for UX Design",
                "category": "visual_design",
                "author": "UXAAG Team"
            }
        },
        {
            "content": """
            Mobile-First Design: With mobile devices dominating web traffic, mobile-first design is essential:
            - Start with mobile layout and scale up to desktop
            - Use touch-friendly interface elements (minimum 44x44 points)
            - Optimize for thumb navigation
            - Ensure fast loading times on mobile networks
            - Test on actual devices, not just simulators
            Mobile-first design improves both mobile and desktop experiences.
            """,
            "metadata": {
                "source": "mobile_design_guide",
                "title": "Mobile-First Design Strategy",
                "category": "responsive_design",
                "author": "UXAAG Team"
            }
        }
    ]
    
    # Process each document
    for doc in sample_docs:
        # Chunk the document
        chunks = chunker.chunk_document(
            content=doc["content"],
            metadata=doc["metadata"]
        )
        
        # Generate embeddings for chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = embedder.embed_documents(chunk_texts)
        
        # Prepare documents for vector database
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "content": chunk.content,
                "metadata": {
                    **chunk.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
        
        # Add to vector database
        success = vector_db.add_documents(
            documents=documents,
            embeddings=embeddings,
            ids=[f"{doc['metadata']['source']}_chunk_{i}" for i in range(len(chunks))]
        )
        
        if success:
            logger.info(f"✓ Added document: {doc['metadata']['title']}")
        else:
            logger.error(f"✗ Failed to add document: {doc['metadata']['title']}")
    
    logger.info("Sample documents added successfully!")

def test_rag_system(rag_engine):
    """
    Test the RAG system with sample queries.
    
    Args:
        rag_engine: RAGEngine instance
    """
    logger.info("Testing RAG system with sample queries...")
    
    test_queries = [
        "What are the key UX design principles?",
        "How should I use colors in UX design?",
        "What is mobile-first design and why is it important?",
        "How can I improve accessibility in my designs?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        try:
            response = rag_engine.generate_response(query)
            print(f"Answer: {response.answer}")
            print(f"Confidence: {response.confidence_score:.2f}")
            print(f"Sources: {', '.join(response.sources)}")
            print(f"Documents retrieved: {len(response.retrieved_documents)}")
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
            print(f"Error: {str(e)}")
    
    print(f"\n{'='*60}")
    logger.info("RAG system testing completed!")

def main():
    """Main function to run the RAG integration example."""
    try:
        # Set up all components
        rag_engine, vector_db, embedder, chunker = setup_rag_components()
        
        # Add sample documents
        add_sample_documents(vector_db, embedder, chunker)
        
        # Test the system
        test_rag_system(rag_engine)
        
        logger.info("RAG integration example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in RAG integration: {str(e)}")
        raise

if __name__ == "__main__":
    main() 