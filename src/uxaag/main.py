from dotenv import load_dotenv
from langchain.agents import tool, AgentExecutor, initialize_agent, AgentType
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain_openai import  AzureOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

@tool
def get_text_length(text : str) -> int:
    """Returns a string containing the text to be processed."""
    return len(text)

if __name__ == "__main__":

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
    
    template = """You are a professional assistant specializing in providing concise, accurate, and actionable information. Use the provided context to answer the question in a clear, structured format suitable for professional use:

    Context: {context}
    Question: {question}

    Response:
    """
    
    prompt = PromptTemplate.from_template(template = template).partial(
        tools = render_text_description(tools), 
        tool_names = ", ".join([tool.name for tool in tools])
    )
    llm = AzureOpenAI(
        azure_endpoint="https://models.inference.ai.azure.com",
        api_key=token,
        api_version="2024-06-01",
        deployment_name="gpt-4o"
    )
    agent = ({"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser())
    # agent_executor = AgentExecutor(
    #     agent=agent,
    #     tools=tools,
    #     llm=llm,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    #     handle_parsing_errors=True
    # ) 
    context = "Cloud computing in 2025 is expected to focus on AI integration, enhanced security, and hybrid cloud solutions."
    question = "What are the key benefits of using cloud computing for enterprises in 2025?"
    try:
        result = agent.invoke({"context": context, "question": question})
        print("Agent Response:")
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")

    