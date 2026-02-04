"""
Web crawler implementation for fetching and parsing web content.
"""

import asyncio
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl, Field
import logging

class CrawlResult(BaseModel):
    """Model for crawl results."""
    url: HttpUrl = Field(..., description="The URL that was crawled")
    content: Dict[str, Any] = Field(..., description="The crawled content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the crawl")
    links: List[str] = Field(default_factory=list, description="Links found on the page")

class WebCrawler:
    """Web crawler for fetching and parsing web content."""
    
    def __init__(
        self,
        max_depth: int = 2,
        max_pages: int = 10,
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None,
        allowed_domains: Optional[List[str]] = None
    ):
        """Initialize the web crawler.
        
        Args:
            max_depth: Maximum depth to crawl
            max_pages: Maximum number of pages to crawl
            timeout: Request timeout in seconds
            headers: Custom HTTP headers
            allowed_domains: List of allowed domains to crawl
        """
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.timeout = timeout
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.allowed_domains = allowed_domains
        self.visited_urls: Set[str] = set()
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Set up the crawler session."""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up the crawler session."""
        if self.session:
            await self.session.close()
            
    def _is_allowed_domain(self, url: str) -> bool:
        """Check if the URL's domain is allowed."""
        if not self.allowed_domains:
            return True
        domain = urlparse(url).netloc
        return any(allowed in domain for allowed in self.allowed_domains)
    
    async def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch a single page's content."""
        if not self.session:
            raise RuntimeError("Crawler session not initialized. Use 'async with' context manager.")
            
        try:
            async with self.session.get(url, timeout=self.timeout) as response:
                if response.status == 200:
                    return await response.text()
                logging.warning(f"Failed to fetch {url}: Status {response.status}")
                return None
        except Exception as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            return None
            
    def _parse_content(self, html: str, base_url: str) -> Dict[str, Any]:
        """Parse HTML content and extract relevant information."""
        soup = BeautifulSoup(html, 'lxml')
        
        # Extract text content
        text_content = soup.get_text(separator=' ', strip=True)
        
        # Extract structured data
        structured_data = {}
        
        # Extract JSON-LD
        json_ld = soup.find_all('script', type='application/ld+json')
        if json_ld:
            structured_data['json-ld'] = [script.string for script in json_ld]
            
        # Extract meta tags
        meta_tags = {}
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                meta_tags[name] = content
                
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            if self._is_allowed_domain(absolute_url):
                links.append(absolute_url)
                
        return {
            'text': text_content,
            'structured_data': structured_data,
            'meta_tags': meta_tags,
            'links': links
        }
        
    async def crawl(
        self,
        start_url: str,
        depth: int = 0,
        visited: Optional[Set[str]] = None
    ) -> List[CrawlResult]:
        """Crawl web pages starting from the given URL.
        
        Args:
            start_url: The URL to start crawling from
            depth: Current crawl depth
            visited: Set of already visited URLs
            
        Returns:
            List of crawl results
        """
        if visited is None:
            visited = set()
            
        if depth > self.max_depth or len(visited) >= self.max_pages:
            return []
            
        if start_url in visited or not self._is_allowed_domain(start_url):
            return []
            
        visited.add(start_url)
        self.visited_urls.add(start_url)
        
        # Fetch and parse the page
        html = await self._fetch_page(start_url)
        if not html:
            return []
            
        content = self._parse_content(html, start_url)
        
        # Create crawl result
        result = CrawlResult(
            url=start_url,
            content=content,
            metadata={
                'depth': depth,
                'crawl_time': asyncio.get_event_loop().time()
            },
            links=content['links']
        )
        
        # Recursively crawl linked pages
        results = [result]
        if depth < self.max_depth:
            tasks = []
            for link in content['links']:
                if link not in visited and len(visited) < self.max_pages:
                    tasks.append(self.crawl(link, depth + 1, visited))
            if tasks:
                sub_results = await asyncio.gather(*tasks)
                for sub_result in sub_results:
                    results.extend(sub_result)
                    
        return results
        
    async def crawl_single(self, url: str) -> Optional[CrawlResult]:
        """Crawl a single URL without following links.
        
        Args:
            url: The URL to crawl
            
        Returns:
            CrawlResult if successful, None otherwise
        """
        if not self._is_allowed_domain(url):
            return None
            
        html = await self._fetch_page(url)
        if not html:
            return None
            
        content = self._parse_content(html, url)
        
        return CrawlResult(
            url=url,
            content=content,
            metadata={
                'crawl_time': asyncio.get_event_loop().time()
            },
            links=content['links']
        ) 