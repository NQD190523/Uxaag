import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import tiktoken

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = AzureChatOpenAI(
    azure_endpoint="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"],
    api_version="2024-04-01-preview",
    model="gpt-4o",
    max_tokens=4000  # Maximum output tokens
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are UXAAG, a UX design AI assistant. Provide concise, actionable UX solutions in bullet points for the given question, using the conversation history for context. For general queries about UXAAG, describe its purpose in 3-5 bullet points. For color-related queries, include specific hex codes. Do not include phrases like "Response end"."""),  # System prompt for instructions
    MessagesPlaceholder(variable_name="history"),  # Placeholder for conversation history
    ("human", "{question}")  # Current user question
])

# Initialize in-memory chat history store
store = {}

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

# Create the runnable with message history
chain = RunnableWithMessageHistory(
    runnable=prompt | llm,
    get_session_history=get_chat_history,
    input_messages_key="question",
    history_messages_key="history"
)

# Function to generate UX response
def generate_ux_response(question: str, session_id: str = "default") -> str:
    try:
        # Invoke the chain with the question
        response = chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        ).content
        return response
    except Exception as e:
        print(f"Error processing '{question}': {str(e)}")
        return f"Error: {str(e)}"

def print_separator():
    print("\n" + "="*80 + "\n")

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
    interactive_session()