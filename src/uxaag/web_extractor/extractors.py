"""
Extractors for processing different types of web data.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

class ExtractionResult(BaseModel):
    """Model for extraction results."""
    content: Any = Field(..., description="The extracted content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the extraction")
    confidence: float = Field(default=1.0, description="Confidence score of the extraction")

class ExtractorTool(BaseTool):
    """Tool wrapper for extractors."""
    
    _extractor: 'BaseExtractor' = PrivateAttr()
    
    def __init__(self, extractor: 'BaseExtractor'):
        """Initialize the tool with an extractor."""
        super().__init__(
            name=extractor.__class__.__name__,
            description=extractor.__doc__,
            func=self._run
        )
        self._extractor = extractor
        
    async def _run(self, data: Any) -> Dict[str, Any]:
        """Run the extractor on the input data."""
        result = await self._extractor.extract(data)
        return result.dict()

class BaseExtractor(ABC):
    """Base class for all extractors."""
    
    @abstractmethod
    async def extract(self, data: Any) -> ExtractionResult:
        """Extract information from the given data."""
        pass
    
    def as_tool(self) -> ExtractorTool:
        """Convert the extractor to a LangChain tool."""
        return ExtractorTool(self)

class TextExtractor(BaseExtractor):
    """Extractor for processing text content from web pages."""
    
    async def extract(self, data: Dict[str, Any]) -> ExtractionResult:
        """Extract text content from web data.
        
        Args:
            data: Dictionary containing web page data with 'text' or 'content' field
            
        Returns:
            ExtractionResult containing the extracted text and metadata
        """
        # Extract text content
        text = data.get('text') or data.get('content', '') 
        
        # Basic text cleaning
        text = text.strip()
        
        return ExtractionResult(
            content=text,
            metadata={
                "type": "text",
                "length": len(text),
                "source": data.get('url', 'unknown')
            }
        )

class DataExtractor(BaseExtractor):
    """Extractor for processing structured data from web pages."""
    
    async def extract(self, data: Dict[str, Any]) -> ExtractionResult:
        """Extract structured data from web content.
        
        Args:
            data: Dictionary containing web page data with structured content
            
        Returns:
            ExtractionResult containing the extracted data and metadata
        """
        # Extract structured data
        structured_data = {}
        
        # Look for common structured data patterns
        if 'json-ld' in data:
            structured_data['json-ld'] = data['json-ld']
        if 'microdata' in data:
            structured_data['microdata'] = data['microdata']
        if 'rdfa' in data:
            structured_data['rdfa'] = data['rdfa']
            
        return ExtractionResult(
            content=structured_data,
            metadata={
                "type": "structured_data",
                "formats": list(structured_data.keys()),
                "source": data.get('url', 'unknown')
            }
        ) 