"""
Prompt Builder for UXAAG RAG Pipeline

This module handles building prompts for the LLM by combining:
1. User queries
2. Retrieved document context
3. Conversation history
4. System instructions

The prompt builder ensures that the LLM receives well-structured input
that maximizes the quality of generated responses.
"""

from typing import List, Dict, Any, Optional
import logging

class PromptBuilder:
    """
    Builds prompts for the LLM by combining query, context, and history.
    
    This class is responsible for creating well-structured prompts that
    help the LLM generate high-quality, contextually relevant responses.
    
    The prompt structure typically includes:
    1. System instructions
    2. Retrieved document context
    3. Conversation history
    4. Current user query
    """
    
    def __init__(
        self,
        system_instructions: str = None,
        max_history_length: int = 5,
        context_format: str = "structured"
    ):
        """
        Initialize the prompt builder.
        
        Args:
            system_instructions: Base system instructions for the LLM
            max_history_length: Maximum number of conversation turns to include
            context_format: How to format the context ("structured", "simple")
        """
        self.system_instructions = system_instructions or self._get_default_instructions()
        self.max_history_length = max_history_length
        self.context_format = context_format
        
        logging.info(f"Initialized PromptBuilder: max_history={max_history_length}, format={context_format}")
    
    def _get_default_instructions(self) -> str:
        """Get default system instructions for UX design assistance."""
        return """You are UXAAG, a UX design AI assistant. Your role is to provide helpful, 
        accurate, and actionable UX design advice based on the context provided. 

        Guidelines:
        - Use the retrieved documents as your primary source of information
        - Provide specific, actionable advice when possible
        - Cite sources when referencing specific information
        - Be concise but comprehensive
        - Focus on practical UX design principles and best practices
        - If the context doesn't contain relevant information, acknowledge this and provide general guidance
        """
    
    def build_prompt(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build a complete prompt for the LLM.
        
        Args:
            query: The user's current question
            context: Retrieved document context
            conversation_history: Previous conversation turns
            
        Returns:
            Formatted prompt string for the LLM
        """
        # Build the prompt components
        prompt_parts = []
        
        # 1. System instructions
        prompt_parts.append(f"System Instructions:\n{self.system_instructions}\n")
        
        # 2. Context from retrieved documents
        if context:
            prompt_parts.append(self._format_context(context))
        
        # 3. Conversation history (limited length)
        if conversation_history:
            history_context = self._format_conversation_history(conversation_history)
            if history_context:
                prompt_parts.append(history_context)
        
        # 4. Current query
        prompt_parts.append(f"User Question: {query}\n")
        
        # 5. Response instructions
        prompt_parts.append("Please provide a helpful response based on the context above.")
        
        return "\n".join(prompt_parts)
    
    def _format_context(self, context: str) -> str:
        """Format the retrieved document context."""
        if self.context_format == "structured":
            return f"Retrieved Documents:\n{context}\n"
        else:
            return f"Context:\n{context}\n"
    
    def _format_conversation_history(
        self, 
        history: List[Dict[str, str]]
    ) -> str:
        """Format conversation history for context."""
        if not history:
            return ""
        
        # Limit history length
        limited_history = history[-self.max_history_length:]
        
        history_parts = ["Previous Conversation:"]
        for i, turn in enumerate(limited_history):
            user_msg = turn.get("user", "")
            ai_msg = turn.get("assistant", "")
            
            if user_msg:
                history_parts.append(f"User: {user_msg}")
            if ai_msg:
                history_parts.append(f"Assistant: {ai_msg}")
        
        return "\n".join(history_parts) + "\n"
    
    def build_simple_prompt(self, query: str, context: str) -> str:
        """
        Build a simple prompt without conversation history.
        
        Args:
            query: The user's question
            context: Retrieved document context
            
        Returns:
            Simple formatted prompt
        """
        return f"""Context: {context}

Question: {query}

Answer:""" 