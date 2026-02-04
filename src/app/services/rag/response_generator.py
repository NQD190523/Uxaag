"""
Response Generator for UXAAG RAG Pipeline

This module handles generating responses from the LLM based on:
1. Constructed prompts with context
2. LLM configuration and parameters
3. Response formatting and validation

The response generator is responsible for:
- Sending prompts to the LLM
- Processing LLM responses
- Handling errors and fallbacks
- Formatting responses for the user
"""

from typing import Dict, Any, Optional
import logging
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage

class ResponseGenerator:
    """
    Generates responses using the language model.
    
    This class handles the interaction with the LLM to generate
    responses based on the constructed prompts. It also handles
    error cases and response formatting.
    
    The response generation process:
    1. Send formatted prompt to LLM
    2. Process LLM response
    3. Handle any errors or edge cases
    4. Return formatted response
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        max_retries: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize the response generator.
        
        Args:
            llm: Language model instance (Azure OpenAI, etc.)
            max_retries: Maximum number of retry attempts
            temperature: LLM temperature for response generation
            max_tokens: Maximum tokens in generated response
        """
        self.llm = llm
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logging.info(f"Initialized ResponseGenerator: max_retries={max_retries}, temp={temperature}")
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The formatted prompt to send to the LLM
            
        Returns:
            Generated response text from the LLM
            
        Raises:
            Exception: If response generation fails after retries
        """
        for attempt in range(self.max_retries):
            try:
                logging.info(f"Generating LLM response (attempt {attempt + 1})")
                
                # Generate response using the LLM
                response = self._invoke_llm(prompt)
                
                if response and response.strip():
                    logging.info("Successfully generated LLM response")
                    return response.strip()
                else:
                    logging.warning(f"Empty response from LLM (attempt {attempt + 1})")
                    
            except Exception as e:
                logging.error(f"Error generating response (attempt {attempt + 1}): {str(e)}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"Failed to generate response after {self.max_retries} attempts: {str(e)}")
                
                # Wait before retrying
                import time
                time.sleep(1)
        
        # Fallback response if all attempts fail
        return self._get_fallback_response()
    
    def _invoke_llm(self, prompt: str) -> str:
        """
        Invoke the language model with the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Response text from the LLM
        """
        try:
            # For Azure OpenAI and similar LangChain models
            if hasattr(self.llm, 'invoke'):
                # Try using the invoke method first
                response = self.llm.invoke(prompt)
                if hasattr(response, 'content'):
                    return response.content
                elif isinstance(response, str):
                    return response
                else:
                    return str(response)
            
            elif hasattr(self.llm, 'generate'):
                # Fallback to generate method
                response = self.llm.generate([prompt])
                if hasattr(response, 'generations') and response.generations:
                    return response.generations[0][0].text
                else:
                    return str(response)
            
            else:
                # Last resort - try calling directly
                return str(self.llm(prompt))
                
        except Exception as e:
            logging.error(f"Error invoking LLM: {str(e)}")
            raise
    
    def _get_fallback_response(self) -> str:
        """Get a fallback response when LLM generation fails."""
        return ("I apologize, but I'm having trouble generating a response right now. "
                "Please try rephrasing your question or ask about a different topic.")
    
    def generate_response_with_messages(
        self, 
        system_message: str, 
        user_message: str
    ) -> str:
        """
        Generate response using message-based format.
        
        Args:
            system_message: System instructions
            user_message: User's question or input
            
        Returns:
            Generated response text
        """
        try:
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            logging.error(f"Error generating response with messages: {str(e)}")
            return self._get_fallback_response()
    
    def validate_response(self, response: str) -> bool:
        """
        Validate that the generated response is acceptable.
        
        Args:
            response: The generated response to validate
            
        Returns:
            True if response is valid, False otherwise
        """
        if not response or not response.strip():
            return False
        
        # Check for common error patterns
        error_patterns = [
            "I apologize, but I cannot",
            "I'm sorry, I don't have access to",
            "I cannot provide",
            "I'm unable to",
            "Error:",
            "Exception:"
        ]
        
        response_lower = response.lower()
        for pattern in error_patterns:
            if pattern.lower() in response_lower:
                return False
        
        # Check minimum length
        if len(response.strip()) < 10:
            return False
        
        return True
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about response generation."""
        return {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "average_response_length": 0,
            "average_generation_time": 0.0
        } 