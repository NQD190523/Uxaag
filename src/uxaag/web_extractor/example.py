"""
Example usage of the web crawler and extractor.
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from uxaag.web_extractor.crawler import WebCrawler
from uxaag.web_extractor.agent import WebExtractorAgent

async def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Initialize the web extractor agent
    extractor = WebExtractorAgent(llm=llm)
    
    # Example URL to crawl
    target_url = "https://example.com"  # Replace with your target URL
    
    print(f"Starting to crawl: {target_url}")
    
    # Crawl the website
    async with WebCrawler(max_depth=1, max_pages=3) as crawler:
        # Crawl a single page first
        result = await crawler.crawl_single(target_url)
        
        if result:
            print(f"\nCrawled URL: {result.url}")
            print(f"Found {len(result.links)} links")
            
            # Extract information using the web extractor
            extraction_result = await extractor.extract(
                web_data=result.content,
                requirements="Extract the main heading and any product information"
            )
            
            print("\nExtracted Information:")
            print(extraction_result["extracted_data"])
            
            # If you want to crawl more pages, use the recursive crawl
            print("\nCrawling linked pages...")
            all_results = await crawler.crawl(target_url)
            
            print(f"\nTotal pages crawled: {len(all_results)}")
            for page in all_results:
                print(f"\nProcessing: {page.url}")
                extraction = await extractor.extract(
                    web_data=page.content,
                    requirements="Extract the main heading and any product information"
                )
                print(f"Extracted: {extraction['extracted_data']}")

if __name__ == "__main__":
    asyncio.run(main()) 