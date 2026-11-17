"""
Web Crawler for the Web Crawler Service.
"""

import asyncio
import aiohttp
from urllib.parse import urljoin, urlparse
from typing import Dict, Any, List, Set
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WebCrawler:
    """Web crawler for extracting content from websites."""
    
    def __init__(self):
        """Initialize the web crawler."""
        self.session = None
        self.visited_urls: Set[str] = set()
        self.crawled_urls: List[str] = []
        self.errors: List[str] = []
    
    async def crawl(
        self,
        url: str,
        max_depth: int = 1,
        max_pages: int = 10,
        follow_links: bool = True
    ) -> Dict[str, Any]:
        """
        Crawl a website and extract content.
        
        Args:
            url: Starting URL to crawl
            max_depth: Maximum crawling depth
            max_pages: Maximum number of pages to crawl
            follow_links: Whether to follow links
            
        Returns:
            Dictionary with crawling results
        """
        try:
            self.visited_urls.clear()
            self.crawled_urls.clear()
            self.errors.clear()
            
            async with aiohttp.ClientSession() as session:
                self.session = session
                await self._crawl_recursive(url, max_depth, max_pages, follow_links)
            
            return {
                'crawled_urls': self.crawled_urls.copy(),
                'total_pages': len(self.crawled_urls),
                'errors': self.errors.copy()
            }
            
        except Exception as e:
            logger.error(f"Crawling failed: {str(e)}")
            raise
    
    async def _crawl_recursive(
        self,
        url: str,
        max_depth: int,
        max_pages: int,
        follow_links: bool,
        current_depth: int = 0
    ) -> None:
        """Recursively crawl URLs."""
        if (current_depth >= max_depth or 
            len(self.crawled_urls) >= max_pages or 
            url in self.visited_urls):
            return
        
        self.visited_urls.add(url)
        
        try:
            # Extract content from current page
            content = await self.extract_page_content(url)
            if content:
                self.crawled_urls.append(url)
                logger.info(f"Crawled page {len(self.crawled_urls)}: {url}")
            
            # Follow links if enabled and within limits
            if (follow_links and 
                current_depth < max_depth - 1 and 
                len(self.crawled_urls) < max_pages):
                
                links = await self._extract_links(url)
                for link in links:
                    if len(self.crawled_urls) >= max_pages:
                        break
                    await self._crawl_recursive(
                        link, max_depth, max_pages, follow_links, current_depth + 1
                    )
                    
        except Exception as e:
            error_msg = f"Failed to crawl {url}: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
    
    async def extract_page_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a single page.
        
        Args:
            url: URL to extract content from
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}")
                
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract text content
                text_content = self._extract_text(soup)
                
                # Extract metadata
                metadata = self._extract_metadata(soup, url)
                
                return {
                    'url': url,
                    'content': text_content,
                    'metadata': metadata,
                    'status_code': response.status
                }
                
        except Exception as e:
            logger.error(f"Failed to extract content from {url}: {str(e)}")
            raise
    
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract text content from BeautifulSoup object."""
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
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract metadata from BeautifulSoup object."""
        metadata = {
            'url': url,
            'title': '',
            'description': '',
            'keywords': '',
            'links_count': 0
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            metadata['description'] = meta_desc.get('content', '').strip()
        
        # Extract meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            metadata['keywords'] = meta_keywords.get('content', '').strip()
        
        # Count links
        links = soup.find_all('a', href=True)
        metadata['links_count'] = len(links)
        
        return metadata
    
    async def _extract_links(self, url: str) -> List[str]:
        """Extract links from a page."""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return []
                
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')
                
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    absolute_url = urljoin(url, href)
                    
                    # Only include HTTP/HTTPS links
                    if absolute_url.startswith(('http://', 'https://')):
                        links.append(absolute_url)
                
                return links
                
        except Exception as e:
            logger.error(f"Failed to extract links from {url}: {str(e)}")
            return []

