"""
Web Data Extraction Agent Module

This module provides functionality for extracting and processing web crawl data.
"""

from .agent import WebExtractorAgent
from .extractors import BaseExtractor, TextExtractor, DataExtractor
from .crawler import WebCrawler, CrawlResult

__all__ = [
    'WebExtractorAgent',
    'BaseExtractor',
    'TextExtractor',
    'DataExtractor',
    'WebCrawler',
    'CrawlResult'
] 