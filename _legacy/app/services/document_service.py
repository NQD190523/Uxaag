"""
Document extractor for the Document Service.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import aiohttp
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


class DocumentExtractResult:
    """Result of document extraction."""
    
    def __init__(self, content: str, metadata: Dict[str, Any], confidence: float):
        self.content = content
        self.metadata = metadata
        self.confidence = confidence


class DocumentExtractor:
    """Document extractor for various file types and URLs."""
    
    def __init__(self):
        self.supported_extensions = {'.txt', '.html', '.htm', '.md', '.pdf', '.docx', '.doc'}
    
    async def extract(self, request_data: Dict[str, Any]) -> DocumentExtractResult:
        """
        Extract content from a document.
        
        Args:
            request_data: Dictionary containing extraction parameters
            
        Returns:
            DocumentExtractResult with extracted content and metadata
        """
        try:
            # Determine extraction method based on input
            if 'path' in request_data and request_data['path']:
                return await self._extract_from_file(request_data['path'])
            elif 'url' in request_data and request_data['url']:
                return await self._extract_from_url(request_data['url'])
            elif 'content' in request_data and request_data['content']:
                return await self._extract_from_content(request_data['content'])
            else:
                raise ValueError("No valid input provided (path, url, or content)")
                
        except Exception as e:
            logger.error(f"Document extraction failed: {str(e)}")
            raise
    
    async def _extract_from_file(self, file_path: str) -> DocumentExtractResult:
        """Extract content from a file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file extension
        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        try:
            # Read file content
            if path.suffix.lower() in {'.txt', '.md'}:
                content = path.read_text(encoding='utf-8')
                confidence = 1.0
            elif path.suffix.lower() in {'.html', '.htm'}:
                content = self._extract_text_from_html(path.read_text(encoding='utf-8'))
                confidence = 0.9
            else:
                # For other file types, we'd need specialized libraries
                # For now, just read as text
                content = path.read_text(encoding='utf-8', errors='ignore')
                confidence = 0.7
            
            metadata = {
                'file_path': str(path),
                'file_name': path.name,
                'file_size': path.stat().st_size,
                'file_extension': path.suffix,
                'extraction_method': 'file'
            }
            
            return DocumentExtractResult(content, metadata, confidence)
            
        except Exception as e:
            logger.error(f"Failed to extract from file {file_path}: {str(e)}")
            raise
    
    async def _extract_from_url(self, url: str) -> DocumentExtractResult:
        """Extract content from a URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to fetch URL: {response.status}")
                    
                    html_content = await response.text()
                    content = self._extract_text_from_html(html_content)
                    
                    metadata = {
                        'url': url,
                        'status_code': response.status,
                        'content_type': response.headers.get('content-type', ''),
                        'extraction_method': 'url'
                    }
                    
                    return DocumentExtractResult(content, metadata, 0.8)
                    
        except Exception as e:
            logger.error(f"Failed to extract from URL {url}: {str(e)}")
            raise
    
    async def _extract_from_content(self, content: str) -> DocumentExtractResult:
        """Extract content from raw text."""
        metadata = {
            'extraction_method': 'content',
            'content_length': len(content)
        }
        
        return DocumentExtractResult(content, metadata, 1.0)
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract text content from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Failed to extract text from HTML: {str(e)}")
            return html_content  # Return raw HTML if parsing fails

