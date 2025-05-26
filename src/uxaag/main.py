import os
import asyncio
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory, RunnableLambda
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_core.tools import tool
import tiktoken
from uxaag.web_extractor.agent import WebExtractorAgent
from uxaag.web_extractor.crawler import WebCrawler

target_url = "https://www.nngroup.com/articles/definition-user-experience/"

def load_environment():
    """Load environment variables and verify required ones are present."""
    load_dotenv(override=True)  # override=True ensures new values take precedence
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        raise ValueError("GITHUB_TOKEN not found in environment variables. Please check your .env file.")
    return github_token

# Load environment variables
github_token = load_environment()
# print(github_token)

# Define a simple function to print a message
@tool
def log_query(query: str) -> None:
    """Prints a message to log the query being processed."""
    print(f"Processing UX query: {query}")


# Initialize the LLM
llm = AzureChatOpenAI(
    api_key=github_token,  # Use the loaded token
    api_version="2024-06-01",
    azure_endpoint="https://models.inference.ai.azure.com",
    model="gpt-4o",
    max_tokens=4000  # Maximum output tokens
)

# Bind the tool to the LLM
tools = [log_query]
llm_with_tools = llm.bind_tools(tools)

# Initialize the web extractor agent
extractor = WebExtractorAgent(llm=llm)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are UXAAG, a UX design AI assistant. Provide concise, actionable UX solutions in bullet points for the given question, using the conversation history for context. For general queries about UXAAG, describe its purpose in 3-5 bullet points. For color-related queries, include specific hex codes. Do not include phrases like "Response end"."""),  # System prompt for instructions
    MessagesPlaceholder(variable_name="history"),  # Placeholder for conversation history
    ("human", "{question}")  # Current user question
])

# Initialize in-memory chat history store
store = {}

async def main():
    # Crawl the website
    async with WebCrawler(max_depth=1, max_pages=3) as crawler:
        # Crawl a single page first
        result = await crawler.crawl_single(target_url)
        
        if result:
            print(f"\nCrawled URL: {result.url}")
            print(f"Found {len(result.links)} links")
            
            # Extract information using the web extractor
            extraction_result = await extractor.extract(
                web_data=result.content,
                requirements="Extract the main heading and any product information"
            )
            print("\nExtracted Information:")
            print(extraction_result["extracted_data"])
            
            # Store the extracted data in the chat history
            if "interactive_session" not in store:
                store["interactive_session"] = InMemoryChatMessageHistory()
            store["interactive_session"].add_message(HumanMessage(content=extraction_result["extracted_data"]))

# Function to estimate token count using tiktoken
def estimate_tokens(messages: list) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4o")
    text = "".join(str(msg.content) for msg in messages)
    return len(encoding.encode(text))

# Function to truncate history to 2000 tokens
def get_chat_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    history = store[session_id]
    #  Get messages and truncate to 2000 tokens
    messages = history.messages
    if not messages:
        return history
    total_tokens = estimate_tokens(messages)
    while total_tokens > 2000 and messages:
        messages.pop(0)  # Remove oldest message
        total_tokens = estimate_tokens(messages)
    history.messages = messages
    return history

# Create a custom chain to handle tool calls and format output
def create_tool_chain():
    def handle_tool_calls(response: AIMessage) -> dict:
        # Handle tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "log_query":
                    log_query.invoke({"query": tool_call["args"]["query"]})
                    # After tool call, get a new response from the LLM
                    messages = prompt.format_messages(
                        question=tool_call["args"]["query"],
                        history=[]
                    )
                    response = llm.invoke(messages)
        
        # Ensure we have content in the response
        if not response.content:
            return {"answer": "I apologize, but I couldn't generate a response. Please try again."}
            
        return {"answer": response.content}
    
    # Create the chain: prompt -> llm_with_tools -> handle_tool_calls
    return prompt | llm_with_tools | RunnableLambda(handle_tool_calls)

# Create the runnable with message history
chain = RunnableWithMessageHistory(
    runnable=create_tool_chain(),
    get_session_history=get_chat_history,
    input_messages_key="question",
    history_messages_key="history",
    output_messages_key="answer"
)

# Function to generate UX response
def generate_ux_response(question: str, session_id: str = "default") -> str:
    try:
        # Invoke the chain with the question
        response = chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        )
        
        # Ensure we have a valid response
        if not response or "answer" not in response:
            return "I apologize, but I couldn't generate a response. Please try again."
            
        return response["answer"]
    except Exception as e:
        print(f"Error processing '{question}': {str(e)}")
        return f"Error: {str(e)}"

# Function to print a separator
def print_separator():
    print("\n" + "="*80 + "\n")

# Function to start an interactive session
def interactive_session():
    print_separator()
    print("Welcome to UXAAG - UX Design AI Assistant!")
    print("Type 'exit' to end the conversation")
    print("Type 'clear' to clear the conversation history")
    print_separator()
    
    session_id = "interactive_session"
    store.clear()  # Clear any existing history
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit command
            if user_input.lower() == "exit":
                print_separator()
                print("Thank you for using UXAAG! Goodbye!")
                print_separator()
                break
                
            # Check for clear command
            if user_input.lower() == "clear":
                store.clear()
                print("\nConversation history cleared!")
                continue
                
            # Skip empty inputs
            if not user_input:
                continue
                
            # Get response from the model
            response = generate_ux_response(user_input, session_id)
            
            # Print the response with formatting
            print("\nUXAAG:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\nExiting UXAAG...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    asyncio.run(main())
    interactive_session()