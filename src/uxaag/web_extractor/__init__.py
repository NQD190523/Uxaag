"""
Web Data Extraction Agent Module

This module provides functionality for extracting and processing web crawl data.
"""

# Handle imports for both direct execution and module import
try:
    # Try relative imports first (when imported as module)
    from .agent import WebExtractorAgent
    from .extractors import BaseExtractor, TextExtractor, DataExtractor
    from .crawler import WebCrawler, CrawlResult
except ImportError:
    # Fall back to absolute imports (when run directly)
    from agent import WebExtractorAgent
    from extractors import BaseExtractor, TextExtractor, DataExtractor
    from crawler import WebCrawler, CrawlResult

__all__ = [
    'WebExtractorAgent',
    'BaseExtractor',
    'TextExtractor',
    'DataExtractor',
    'WebCrawler',
    'CrawlResult'
] 