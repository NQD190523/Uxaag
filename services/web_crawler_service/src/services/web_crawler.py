"""Async web crawler (service-style helper in the router package).

NOTE: not wired up. Nothing imports this module because the router that would use it is not
mounted. Implements a depth-first, same-domain-by-default crawl with aiohttp + BeautifulSoup.
It does not honor robots.txt and has no rate limiting or SSRF protection (any URL, including
internal addresses, would be fetched).
"""

import logging
import asyncio
from typing import List, Dict, Any, Set, Optional
from urllib.parse import urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WebCrawler:
    """Async recursive crawler.

    State (``visited_urls``, ``crawled_urls``, ``errors``) lives on the instance and is reset at
    the start of each ``crawl``, so one instance must not run concurrent crawls.
    """
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """Args:
            timeout: Total timeout per HTTP request, in seconds.
            max_retries: Attempts per URL before giving up.
        """
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.visited_urls: Set[str] = set()
        self.crawled_urls: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
    
    async def crawl(
        self,
        start_url: str,
        max_depth: int = 2,
        max_pages: int = 50,
        follow_links: bool = True,
        same_domain_only: bool = True
    ) -> Dict[str, Any]:
        """Crawl from ``start_url`` and return the collected pages.

        Args:
            start_url: Starting URL.
            max_depth: Maximum link depth (0 = start page only).
            max_pages: Stop after this many pages were stored.
            follow_links: Whether to follow links found on pages.
            same_domain_only: Restrict links to the start URL's host (netloc).

        Returns:
            Dict with ``start_url``, ``pages_crawled``, ``pages``, ``errors`` and ``status``
            (``completed`` or ``failed``; ``total_visited`` is only present on success). Exceptions
            are caught and reported as ``failed`` rather than raised.
        """
        try:
            self.visited_urls.clear()
            self.crawled_urls.clear()
            self.errors.clear()
            
            # Compared against each link's netloc to implement ``same_domain_only``.
            base_domain = urlparse(start_url).netloc
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                await self._crawl_recursive(
                    session=session,
                    url=start_url,
                    depth=0,
                    max_depth=max_depth,
                    max_pages=max_pages,
                    follow_links=follow_links,
                    same_domain_only=same_domain_only,
                    base_domain=base_domain
                )
            
            return {
                "start_url": start_url,
                "pages_crawled": len(self.crawled_urls),
                "pages": self.crawled_urls,
                "errors": self.errors,
                "total_visited": len(self.visited_urls),
                "status": "completed"
            }
        
        except Exception as e:
            logger.error(f"Crawling failed for {start_url}: {str(e)}")
            return {
                "start_url": start_url,
                "pages_crawled": len(self.crawled_urls),
                "pages": self.crawled_urls,
                "errors": self.errors + [{"url": start_url, "error": str(e)}],
                "status": "failed"
            }
    
    async def _crawl_recursive(
        self,
        session: aiohttp.ClientSession,
        url: str,
        depth: int,
        max_depth: int,
        max_pages: int,
        follow_links: bool,
        same_domain_only: bool,
        base_domain: str
    ):
        """Visit one URL, store its page and recurse into its links (depth-first).

        Stops when the URL was visited, the page limit is reached or the depth limit is exceeded.
        Per-page failures are recorded in ``self.errors`` and do not abort the crawl.
        """
        # Check if we've already visited this URL
        if url in self.visited_urls:
            return
        
        # Check page limit
        if len(self.crawled_urls) >= max_pages:
            return
        
        # Check depth limit
        if depth > max_depth:
            return
        
        self.visited_urls.add(url)
        
        logger.info(f"Crawling: {url} (depth: {depth})")
        
        try:
            # Extract page content
            page_data = await self.extract_page_content(session, url)
            
            if page_data:
                page_data["depth"] = depth
                self.crawled_urls.append(page_data)
                
                # Follow links if enabled
                if follow_links and depth < max_depth:
                    links = self._extract_links(
                        page_data["html"],
                        url,
                        same_domain_only,
                        base_domain
                    )
                    
                    for link in links:
                        if len(self.crawled_urls) < max_pages and link not in self.visited_urls:
                            await self._crawl_recursive(
                                session=session,
                                url=link,
                                depth=depth + 1,
                                max_depth=max_depth,
                                max_pages=max_pages,
                                follow_links=follow_links,
                                same_domain_only=same_domain_only,
                                base_domain=base_domain
                            )
        
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            self.errors.append({"url": url, "error": str(e), "depth": depth})
    
    async def extract_page_content(
        self,
        session: aiohttp.ClientSession,
        url: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch one page with retries and extract its text and metadata.

        Args:
            session: Shared aiohttp session (carries the timeout).
            url: URL to fetch.

        Returns:
            Dict with url, title, description, content, raw ``html``, status_code and content_type,
            or None if the page was missing (404, not retried) or all attempts failed.
        """
        for attempt in range(self.max_retries):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract title
                        title_tag = soup.find('title')
                        title = title_tag.get_text() if title_tag else "No title"
                        
                        # Extract description
                        meta_desc = soup.find('meta', attrs={'name': 'description'})
                        description = meta_desc.get('content', '') if meta_desc else ''
                        
                        # Extract main text
                        text_content = soup.get_text(separator='\n', strip=True)
                        
                        return {
                            "url": url,
                            "title": title,
                            "description": description,
                            "content": text_content,
                            "html": html,
                            "status_code": response.status,
                            "content_type": response.content_type
                        }
                    
                    elif response.status == 404:
                        logger.warning(f"Page not found: {url}")
                        return None
                    
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        if attempt < self.max_retries - 1:
                            continue
                        return None
            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching {url}, attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    # Back off longer after a timeout than after other errors (1 s).
                    await asyncio.sleep(2)
                    continue
                self.errors.append({"url": url, "error": "Timeout"})
                return None
            
            except Exception as e:
                logger.warning(f"Error fetching {url}, attempt {attempt + 1}/{self.max_retries}: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                self.errors.append({"url": url, "error": str(e)})
                return None
        
        return None
    
    def _extract_links(
        self,
        html: str,
        base_url: str,
        same_domain_only: bool = True,
        base_domain: str = ""
    ) -> List[str]:
        """Extract absolute, de-duplicated links from HTML.

        Args:
            html: Page HTML.
            base_url: Base for resolving relative links.
            same_domain_only: Keep only links whose host equals ``base_domain``.
            base_domain: Host (netloc) to match.

        Returns:
            At most 50 URLs (fragments removed); empty list on parse errors.
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            links = []
            
            for link_tag in soup.find_all('a', href=True):
                href = link_tag['href']
                
                # Skip certain link types
                if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                    continue
                
                # Convert relative to absolute URLs
                absolute_url = urljoin(base_url, href)
                
                # Filter by domain if needed
                if same_domain_only:
                    link_domain = urlparse(absolute_url).netloc
                    if link_domain != base_domain:
                        continue
                
                # Clean up URL (remove fragments)
                clean_url = absolute_url.split('#')[0]
                
                if clean_url and clean_url not in links:
                    links.append(clean_url)
            
            return links[:50]  # Limit to 50 links per page
        
        except Exception as e:
            logger.error(f"Failed to extract links from {base_url}: {str(e)}")
            return []
    
    async def clear_state(self):
        """Reset visited URLs, collected pages and errors."""
        self.visited_urls.clear()
        self.crawled_urls.clear()
        self.errors.clear()
        logger.info("Crawler state cleared")
