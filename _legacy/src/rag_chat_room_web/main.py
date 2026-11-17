from dotenv import load_dotenv
from langchain.agents import tool, AgentExecutor, initialize_agent, AgentType
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain_openai import  AzureOpenAI, AzureChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
import os

# Load environment variables
load_dotenv()

@tool
def get_text_length(text : str) -> int:
    """Returns a string containing the text to be processed."""
    return len(text)

# Set the environment variable for the OpenAI API key
token = os.getenv("GITHUB_TOKEN")
endpoint = "https://models.inference.ai.azure.com"
tools = [
Tool(
    name="GetTextLength",
    func=get_text_length,
    description="Calculates the length of a given text string"
)
] 
template = """You are RAG Chat Room Web, an AI assistant for collaborative chat rooms with document and link context. Provide concise, actionable answers in bullet points, using the conversation history for context. For tool-related questions (e.g., text length), use the available tools. Do not repeat the history, instructions, or phrases like "Response end." For general queries about RAG Chat Room Web, describe its purpose briefly in bullet points.

Conversation History:
{context}

Current Question: {question}

Tools: {tools}

Response (bullet points only):
"""

prompt = PromptTemplate.from_template(template = template).partial(
    tools = render_text_description(tools), 
    tool_names = ", ".join([tool.name for tool in tools])
)
llm = AzureOpenAI(
    azure_endpoint="https://models.inference.ai.azure.com",
    api_key=token,
    api_version="2024-06-01",
    deployment_name="gpt-4o",
    max_tokens=2000
)
# Initialize memory to store conversation history
memory = ConversationBufferMemory(
    memory_key="context",  # Maps to {context} in PromptTemplate
    input_key="question",  # Maps to input question
    output_key="response",  # Maps to LLM response
    max_token_limit=2000
)
# Function to load memory into the context
def get_memory_context(inputs):
    # Load the conversation history from memory
    memory_context = memory.load_memory_variables({"question": inputs["question"]})
    return {
        "context": memory_context["context"],
        "question": inputs["question"]
    }

# Function to invoke the agent and save the response to memory
# Function to invoke the agent and save the response to memory
def invoke_agent_with_memory(question):
    try:
        result = agent.run(question)
        memory.save_context(
            inputs={"question": question},
            outputs={"response": result}
        )
        return result
    except Exception as e:
        return f"Error: {str(e)}"

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    queries = [
        "What are best practices for managing AI chat rooms with shared documents?",
        "How can I improve the navigation issues mentioned earlier?"
    ]
    for question in queries:
        result = invoke_agent_with_memory(question)
        print(f"\nQuestion: {question}")
        print("Agent Response:")
        print(result)
        print("Response end.")

    
