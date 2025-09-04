"""
UXAAG - UX Design AI Assistant

Main application file that orchestrates the RAG system, web extraction,
and interactive chat functionality.
"""

import logging
import os
import asyncio
from typing import Dict, Any
from pathlib import Path

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory, RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Local imports
from dotenv import load_dotenv

# Add src to Python path for imports
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import modules using absolute imports
from uxaag.knowledge_base.retriever import Retriever
from uxaag.rag_pipeline.prompt_builder import PromptBuilder
from uxaag.rag_pipeline.rag_engine import RAGEngine
from uxaag.rag_pipeline.response_generator import ResponseGenerator
from uxaag.vector_store.chunker import DocumentChunker
from uxaag.vector_store.embedder import DocumentEmbedder
from uxaag.vector_store.vector_db import VectorDatabase
from uxaag.web_extractor.agent import WebExtractorAgent
from uxaag.web_extractor.crawler import WebCrawler

# Configuration
TARGET_URL = "https://www.nngroup.com/articles/ai-creative-teammate/"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global stores
store: Dict[str, ChatMessageHistory] = {}
extracted_data_store: Dict[str, Dict[str, Any]] = {}

# Global RAG components (will be initialized in setup_rag_components)
vector_db = None
embedder = None
chunker = None

def     setup_rag_components():
    """
    Set up all RAG components and wire them together.
    
    Returns:
        Tuple of (rag_engine, vector_db, embedder, chunker, llm)
    """
    logger.info("Setting up RAG components...")
    
    # 1. Initialize Document Embedder first (to get embedding dimensions)
    local_embedder = DocumentEmbedder(
        model_name="all-MiniLM-L6-v2",  # Fast, good quality
        device="cpu"  # Use "cuda" if you have GPU
    )
    logger.info("✓ Document Embedder initialized")
    
    # 2. Initialize Vector Database with correct embedding dimensions
    # Get the project root directory (two levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    vector_store_path = project_root / "data" / "vector_store"
    
    local_vector_db = VectorDatabase(
        persist_directory=str(vector_store_path),
        collection_name="uxaag_documents",
        embedding_dimensions=local_embedder.embedding_dimensions
    )
    logger.info("✓ Vector Database initialized")
    
    # 3. Initialize Document Chunker
    local_chunker = DocumentChunker(
        chunk_size=500,
        chunk_overlap=50,
        chunk_strategy="sentence"
    )
    logger.info("✓ Document Chunker initialized")
    
    # 4. Initialize Retriever
    retriever = Retriever(
        vector_db=local_vector_db,
        embedder=local_embedder,
        top_k=5,
        similarity_threshold=0.5
    )
    logger.info("✓ Retriever initialized")
    
    # 5. Initialize Prompt Builder with enhanced configuration
    prompt_builder = PromptBuilder(
        system_instructions="""You are UXAAG, a UX design AI assistant with expertise in user experience design, usability, and interface design.

        Your capabilities include:
        - Providing actionable UX advice and best practices
        - Analyzing design problems and suggesting solutions
        - Explaining UX concepts with real-world examples
        - Offering guidance on accessibility and user research
        - Suggesting design patterns and interaction models

        Always provide:
        1. Clear, actionable recommendations
        2. Specific examples when possible
        3. Consideration for different user types and contexts
        4. Accessibility and usability best practices
        5. Use bullet points for the response
        6. Include the source of the information in the response""",
        max_history_length=5,
        context_format="structured"
    )
    logger.info("✓ Enhanced Prompt Builder initialized")
    
    # 6. Initialize Response Generator with LLM
    config = load_environment()
    print(f"Using GitHub token: {config['github_token'][:10]}...")
    
    print("Initializing ChatOpenAI with GitHub Models...")
    
    # Use ChatOpenAI with GitHub Models endpoint
    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=config['github_token'],
        openai_api_base="https://models.github.ai/inference/v1",
        temperature=0.7,
        max_tokens=4000,
        max_retries=2
    )
    
    print("✅ LLM initialized with ChatOpenAI (GitHub Models)")
    
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
        llm=response_generator,
        prompt_builder=prompt_builder,
        max_context_length=4000,
        top_k_retrieval=5
    )
    logger.info("✓ RAG Engine initialized")
    
    # Check if vector database is empty and add sample documents if needed
    collection_stats = local_vector_db.get_collection_stats()
    if collection_stats.get('total_documents', 0) == 0:
        logger.info("Vector database is empty. Adding sample UX design documents...")
        add_sample_documents_to_vector_db(local_vector_db, local_embedder, local_chunker)
        logger.info("✓ Sample documents added to vector database")
    
    # Set global variables for other functions to use
    global vector_db, embedder, chunker
    vector_db = local_vector_db
    embedder = local_embedder
    chunker = local_chunker
    
    return rag_engine, local_vector_db, local_embedder, local_chunker, llm

def add_documents_to_vector_db(docs, vector_db, embedder, chunker):
    """
    Add documents to the vector database.
    
    Args:
        docs: List of documents with content and metadata
        vector_db: VectorDatabase instance
        embedder: DocumentEmbedder instance
        chunker: DocumentChunker instance
    """
    logger.info("Adding documents to the knowledge base...")

    for doc in docs:
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
    
    logger.info("Documents added successfully!")

def add_sample_documents_to_vector_db(vector_db, embedder, chunker):
    """
    Add sample UX design documents to the vector database.
    
    Args:
        vector_db: VectorDatabase instance
        embedder: DocumentEmbedder instance
        chunker: DocumentChunker instance
    """
    logger.info("Adding sample UX design documents to the knowledge base...")
    
    sample_docs = [
        {
            "content": """
            User Experience (UX) Design Principles
            
            UX design is the process of creating products that provide meaningful and relevant experiences to users. The key principles include:
            
            1. User-Centered Design: Always design with the user in mind. Understand their needs, goals, and pain points.
            2. Usability: Make interfaces intuitive and easy to use. Users should be able to accomplish their goals with minimal effort.
            3. Accessibility: Design for all users, including those with disabilities. Follow WCAG guidelines for inclusive design.
            4. Consistency: Maintain consistent design patterns, terminology, and visual elements throughout the interface.
            5. Feedback: Provide clear feedback for user actions, both positive and negative.
            6. Error Prevention: Design interfaces that prevent errors before they occur.
            7. Recognition Over Recall: Use familiar patterns and icons that users can recognize instantly.
            8. Flexibility: Allow users to customize their experience and provide multiple ways to accomplish tasks.
            
            These principles form the foundation of effective UX design and should guide every design decision.
            """,
            "metadata": {
                "source": "ux_design_principles",
                "title": "UX Design Principles",
                "category": "design_principles",
                "author": "UXAAG",
                "url": "internal"
            }
        },
        {
            "content": """
            UX Design Best Practices for Web Applications
            
            When designing web applications, consider these best practices:
            
            Navigation Design:
            - Use clear, descriptive navigation labels
            - Implement breadcrumbs for complex hierarchies
            - Ensure navigation is consistent across all pages
            - Provide search functionality for large sites
            
            Form Design:
            - Keep forms short and focused
            - Use inline validation with helpful error messages
            - Group related fields logically
            - Provide progress indicators for multi-step forms
            
            Visual Hierarchy:
            - Use typography to establish clear information hierarchy
            - Implement consistent spacing and alignment
            - Use color strategically to guide attention
            - Ensure sufficient contrast for readability
            
            Mobile Responsiveness:
            - Design for mobile-first approach
            - Use touch-friendly button sizes (minimum 44px)
            - Implement responsive breakpoints
            - Test on various devices and screen sizes
            
            Performance:
            - Optimize images and assets
            - Minimize loading times
            - Implement progressive enhancement
            - Use lazy loading for content
            """,
            "metadata": {
                "source": "web_ux_best_practices",
                "title": "Web UX Best Practices",
                "category": "best_practices",
                "author": "UXAAG",
                "url": "internal"
            }
        },
        {
            "content": """
            Color Theory in UX Design
            
            Color plays a crucial role in user experience design:
            
            Primary Colors:
            - Blue (#007BFF): Trust, stability, professionalism
            - Green (#28A745): Success, growth, nature
            - Red (#DC3545): Error, danger, urgency
            - Yellow (#FFC107): Warning, attention, optimism
            
            Color Psychology:
            - Warm colors (red, orange, yellow) create energy and excitement
            - Cool colors (blue, green, purple) promote calmness and trust
            - Neutral colors (gray, white, black) provide balance and sophistication
            
            Accessibility Considerations:
            - Ensure sufficient color contrast (minimum 4.5:1 for normal text)
            - Don't rely solely on color to convey information
            - Test designs with colorblind users
            - Use color to enhance, not replace, other design elements
            
            Brand Integration:
            - Use brand colors consistently across all touchpoints
            - Create color palettes that work well together
            - Consider cultural color associations
            - Maintain color consistency in different lighting conditions
            """,
            "metadata": {
                "source": "color_theory_ux",
                "title": "Color Theory in UX Design",
                "category": "visual_design",
                "author": "UXAAG",
                "url": "internal"
            }
        }
    ]
    
    add_documents_to_vector_db(sample_docs, vector_db, embedder, chunker)
    logger.info("Sample documents added successfully!")

def load_environment():
    """Load environment variables and verify required ones are present."""
    load_dotenv(override=True)
    
    # Get GitHub token from environment
    github_token = os.getenv('GITHUB_TOKEN')
    
    # If not found, try to get it from user input
    if not github_token:
        print("⚠️  GITHUB_TOKEN not found in environment variables.")
        print("Please enter your GitHub token:")
        github_token = input("GitHub Token: ").strip()
        
    if not github_token:
        raise ValueError("GitHub token is required.")
    
    return {
        'github_token': github_token
    }

def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation)."""
    # Rough approximation: 1 token ≈ 4 characters
    return len(text) // 4

def get_chat_history(session_id: str):
    """Get chat history for a session, truncating to 2000 tokens."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    history = store[session_id]
    
    messages = history.messages
    if not messages:
        return history
        
    total_tokens = estimate_tokens("".join(str(msg.content) for msg in messages))
    while total_tokens > 2000 and messages:
        messages.pop(0)  # Remove oldest message
        total_tokens = estimate_tokens("".join(str(msg.content) for msg in messages))
    history.messages = messages
    return history

# Global variables for RAG components (will be initialized when needed)
rag_engine = None
llm = None
extractor = None

# Define the enhanced prompt template for chat interactions
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are UXAAG, a UX design AI assistant. You have access to:

    1. **Conversation History**: Previous messages in this chat session
    2. **Extracted Web Data**: Recent web content that may be relevant
    3. **UX Knowledge Base**: Your built-in UX expertise and best practices

    **Response Guidelines:**
    - Provide concise, actionable UX solutions in bullet points
    - If web data is relevant, reference it clearly (e.g., "Based on [source]...")
    - Combine web insights with your UX knowledge for comprehensive answers
    - For general UXAAG questions, describe its purpose in 3-5 bullet points
    - For color-related queries, include specific hex codes
    - Always consider accessibility and usability best practices
    - Keep responses focused and practical"""),
    
    MessagesPlaceholder(variable_name="history"),
    
    ("human", "Question: {question}"),
    
    ("system", """**Available Context:**
        Web Data: {extracted_data}
        Additional Context: Use this information to enhance your response when relevant.""")
])

async def crawl_and_extract_data():
    """Crawl website and extract data for the knowledge base."""
    global extractor, vector_db, embedder, chunker
    
    # Initialize components if not already done
    if extractor is None:
        if llm is None:
            _, _, _, _, local_llm = setup_rag_components()
            llm = local_llm
        extractor = WebExtractorAgent(llm=llm)
    
    if vector_db is None or embedder is None or chunker is None:
        _, local_vector_db, local_embedder, local_chunker, _ = setup_rag_components()
        vector_db = local_vector_db
        embedder = local_embedder
        chunker = local_chunker
    
    logger.info("Starting web crawling and data extraction...")
    
    async with WebCrawler(max_depth=1, max_pages=3) as crawler:
        result = await crawler.crawl_single(TARGET_URL)
        
        if result:
            print(f"\nCrawled URL: {result.url}")
            print(f"Found {len(result.links)} links")
            
            # Extract information using the web extractor
            extraction_result = await extractor.extract(
                web_data=result.content,
                requirements="Extract the main content or information related to the heading of the page"
            )
            print("\nExtracted Information:")
            print(extraction_result["extracted_data"])
            
            extracted_text = str(extraction_result["extracted_data"])
            if estimate_tokens(extracted_text) > 4000:
                extracted_text = extracted_text[:16000]  # ~4000 tokens
                
            extracted_data_store["interactive_session"] = {"extracted_data": extracted_text}
            
            # Store the extracted data in the chat history
            if "interactive_session" not in store:
                store["interactive_session"] = ChatMessageHistory() 
            store["interactive_session"].add_message(HumanMessage(content=extraction_result["extracted_data"]))
            
            # Store in vector database
            try:
                web_doc = {
                    "content": extracted_text,
                    "metadata": {
                        "source": TARGET_URL,
                        "title": f"Web Content from {TARGET_URL}",
                        "category": "web_extracted",
                        "author": "Web Crawler",
                        "url": TARGET_URL
                    }
                }
                
                add_documents_to_vector_db([web_doc], vector_db, embedder, chunker)
                print(f"✓ Web content stored in vector database")
                
            except Exception as e:
                logger.error(f"Failed to store web content in vector DB: {str(e)}")
                print(f"⚠ Could not store in vector database: {str(e)}")
            
            logger.info("Web crawling and data extraction completed successfully!")
            return extraction_result["extracted_data"]
    
    logger.warning("No data was extracted from web crawling")
    return None

def create_simple_chain():
    """Create a simplified chain without tool calls."""
    def preprocess_input(input_dict: dict) -> dict:
        global vector_db, embedder, llm
        
        # Initialize components if not already done
        if vector_db is None or embedder is None:
            _, local_vector_db, local_embedder, _, _ = setup_rag_components()
            vector_db = local_vector_db
            embedder = local_embedder
        
        if llm is None:
            _, _, _, _, local_llm = setup_rag_components()
            llm = local_llm
        
        session_id = input_dict.get("session_id", "default")
        extracted_data = extracted_data_store.get(session_id, {}).get("extracted_data", "No extracted data available")
        
        # Use RAG to retrieve relevant documents
        retrieved_docs = vector_db.search(embedder.embed_query(input_dict.get("question", "")), top_k=5)
        extracted_data = "\n".join([doc["page_content"] for doc in retrieved_docs]) if retrieved_docs else "No relevant data found."
        
        if isinstance(extracted_data, dict):
            extracted_data = str(extracted_data.get("extracted_data", "No extracted data available"))
            
        return {
            "question": input_dict.get("question", ""),
            "history": input_dict.get("history", []),
            "extracted_data": extracted_data,
            "session_id": session_id
        }

    return RunnableLambda(preprocess_input) | prompt | llm | (lambda x: {"answer": x.content})

# Create the runnable with message history (will be initialized when needed)
chain = None

def generate_ux_response(question: str, session_id: str = "default") -> str:
    """Generate a UX response using the chat chain."""
    global chain
    
    try:
        history = get_chat_history(session_id)
        
        # Get the extracted data from the history if available
        extracted_data = "No extracted data available"
        if history.messages:
            for msg in reversed(history.messages):
                if isinstance(msg, HumanMessage):
                    extracted_data = msg.content
                    break
        
        # Create the chain if not already created
        if chain is None:
            chain = RunnableWithMessageHistory(
                runnable=create_simple_chain(),
                get_session_history=get_chat_history,
                input_messages_key="question",
                history_messages_key="history",
                output_messages_key="answer"
            )
        
        response = chain.invoke(
            {
                "question": question,
                "extracted_data": extracted_data,
                "session_id": session_id
            },
            config={"configurable": {"session_id": session_id}}
        )
        
        if not response or "answer" not in response:
            return "I apologize, but I couldn't generate a response. Please try again."
            
        return response["answer"]
    except Exception as e:
        print(f"Error processing '{question}': {str(e)}")
        return f"Error: {str(e)}"

def generate_rag_response(question: str) -> str:
    """Generate a response using the RAG engine with PromptBuilder."""
    global rag_engine
    
    try:
        # Initialize RAG components if not already done
        if rag_engine is None:
            rag_engine, _, _, _, _ = setup_rag_components()
        
        response = rag_engine.generate_response(question)
        
        print(f"\n{'='*60}")
        print("RAG Response (using PromptBuilder):")
        print(f"{'='*60}")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence_score:.2f}")
        print(f"Sources: {', '.join(response.sources)}")
        print(f"Documents retrieved: {len(response.retrieved_documents)}")
        
        return response.answer
        
    except Exception as e:
        logger.error(f"Error generating RAG response: {str(e)}")
        return f"I apologize, but I encountered an error: {str(e)}"

def generate_chat_response(question: str, session_id: str = "default") -> str:
    """Generate a response using ChatPromptTemplate for interactive chat."""
    global chain
    
    try:
        history = get_chat_history(session_id)
        
        extracted_data = "No extracted_data available"
        if history.messages:
            for msg in reversed(history.messages):
                if isinstance(msg, HumanMessage):
                    extracted_data = msg.content
                    break
        
        # Create the chain if not already created
        if chain is None:
            chain = RunnableWithMessageHistory(
                runnable=create_simple_chain(),
                get_session_history=get_chat_history,
                input_messages_key="question",
                history_messages_key="history",
                output_messages_key="answer"
            )
        
        response = chain.invoke(
            {
                "question": question,
                "extracted_data": extracted_data,
                "session_id": session_id
            },
            config={"configurable": {"session_id": session_id}}
        )
        
        if not response or "answer" not in response:
            return "I apologize, but I couldn't generate a response. Please try again."
            
        return response["answer"]
    except Exception as e:
        print(f"Error processing '{question}': {str(e)}")
        return f"Error: {str(e)}"

def print_separator():
    """Print a separator line."""
    print("\n" + "="*80 + "\n")

def interactive_session():
    """Start an interactive session with the UXAAG system."""
    print_separator()
    print("Welcome to UXAAG - UX Design AI Assistant!")
    print("Type 'exit' to end the conversation")
    print("Type 'clear' to clear the conversation history")
    print("Type 'crawl' to crawl and extract web data")
    print("Type 'samples' to add sample UX documents to the knowledge base")
    print("Type 'status' to check vector database status")
    print("Type 'rag' to test RAG responses (using PromptBuilder)")
    print("Type 'chat' to test chat responses (using ChatPromptTemplate)")
    print_separator()
    
    session_id = "interactive_session"
    store.clear()
    extracted_data_store.clear()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == "exit":
                print_separator()
                print("Thank you for using UXAAG! Goodbye!")
                print_separator()
                break
                
            if user_input.lower() == "clear":
                store.clear()
                print("\nConversation history cleared!")
                continue
                
            if user_input.lower() == "crawl":
                print("\nStarting web crawling and data extraction...")
                asyncio.run(crawl_and_extract_data())
                continue
                
            if user_input.lower() == "samples":
                print("\nAdding sample UX design documents to the knowledge base...")
                if vector_db and embedder and chunker:
                    add_sample_documents_to_vector_db(vector_db, embedder, chunker)
                    print("✓ Sample documents added successfully!")
                else:
                    print("⚠️  RAG components not initialized. Please wait...")
                continue
                
            if user_input.lower() == "status":
                print("\nChecking vector database status...")
                if vector_db:
                    stats = vector_db.get_collection_stats()
                    print(f"📊 Collection: {stats.get('collection_name', 'Unknown')}")
                    print(f"📄 Total Documents: {stats.get('total_documents', 0)}")
                    print(f"🔢 Embedding Dimensions: {stats.get('embedding_dimensions', 'Unknown')}")
                else:
                    print("⚠️  Vector database not initialized")
                continue
                
            if user_input.lower() == "rag":
                print("\nEnter your question for RAG-based response (using PromptBuilder):")
                rag_question = input("RAG Question: ").strip()
                if rag_question:
                    generate_rag_response(rag_question)
                continue
                
            if user_input.lower() == "chat":
                print("\nEnter your question for chat-based response (using ChatPromptTemplate):")
                chat_question = input("Chat Question: ").strip()
                if chat_question:
                    response = generate_chat_response(chat_question, session_id)
                    print(f"\nChat Response: {response}")
                continue
                
            if not user_input:
                continue
                
            response = generate_ux_response(user_input, session_id)
            
            print("\nUXAAG:")
            print("-" * 40)
            print(store.get(session_id).messages)
            print(response)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\nExiting UXAAG...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again or type 'exit' to quit.")

def main():
    """Main function to run the UXAAG system."""
    try:
        # Initialize RAG components first
        print("🚀 Initializing RAG components...")
        global rag_engine, vector_db, embedder, chunker, llm, extractor
        
        rag_engine, vector_db, embedder, chunker, llm = setup_rag_components()
        extractor = WebExtractorAgent(llm=llm)
        print("✅ RAG components initialized!")
        
        interactive_session()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()