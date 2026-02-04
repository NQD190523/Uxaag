"""
Web Extractor Agent implementation.
"""

from typing import Any, Dict, List, Optional
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.messages import AIMessage, HumanMessage

# Handle imports for both direct execution and module import
try:
    # Try relative imports first (when imported as module)
    from .extractors import BaseExtractor, TextExtractor, DataExtractor
except ImportError:
    # Fall back to absolute imports (when run directly)
    from extractors import BaseExtractor, TextExtractor, DataExtractor

class WebExtractorAgent:
    """Agent responsible for extracting information from web crawl data."""
    
    def __init__(
        self,
        llm: Any,
        extractors: Optional[List[BaseExtractor]] = None,
        verbose: bool = False
    ):
        """Initialize the Web Extractor Agent.
        
        Args:
            llm: The language model to use
            extractors: List of extractors to use for data extraction
            verbose: Whether to enable verbose logging
        """
        self.llm = llm
        self.extractors = extractors or [TextExtractor(), DataExtractor()]
        self.verbose = verbose
        
        # Initialize tools from extractors
        self.tools = [extractor.as_tool() for extractor in self.extractors]
        
        # Create the agent
        self.agent = self._create_agent()
        
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor with tools and prompt."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a web data extraction agent. Your task is to extract relevant information 
            from web crawl data based on user requirements. You have access to various extraction tools 
            that can help you process the data.
            
            When given web crawl data:
            1. Analyze the data structure and content
            2. Use appropriate extraction tools based on the data type
            3. Extract the requested information
            4. Return the results in a structured format
            
            Always explain your extraction process and reasoning."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | self.llm
            | OpenAIFunctionsAgentOutputParser()
        )
        
        # Convert our custom tools to LangChain tool format
        langchain_tools = []
        for tool in self.tools:
            from langchain.tools import Tool
            langchain_tools.append(Tool(
                name=tool.name,
                description=tool.description,
                func=tool.run
            ))
        
        return AgentExecutor(
            agent=agent,
            tools=langchain_tools,
            verbose=self.verbose,
            handle_parsing_errors=True
        )
    
    async def extract(
        self,
        web_data: Dict[str, Any],
        requirements: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Extract information from web data based on requirements.
        
        Args:
            web_data: The raw web crawl data to process
            requirements: User requirements for what information to extract
            chat_history: Optional chat history for context
            
        Returns:
            Dict containing extracted information and metadata
        """
        # Convert chat history to LangChain message format
        messages = []
        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
        
        # Prepare input for the agent
        input_data = {
            "input": f"Web Data: {web_data}\nRequirements: {requirements}",
            "chat_history": messages
        }
        
        # Run the agent
        result = await self.agent.ainvoke(input_data)
        
        return {
            "extracted_data": result["output"],
            "metadata": {
                "extraction_method": result.get("intermediate_steps", []),
                "confidence": result.get("confidence", 1.0)
            }
        } 