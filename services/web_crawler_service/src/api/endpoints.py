"""Web crawler HTTP endpoints (router layer), prefix ``/crawl``.

NOTE: not wired up. ``main.py`` never mounts this router; the crawler lives in
``services/web_crawler.py``. This stub service has no ``pyproject.toml`` yet, so the
``shared`` lib must be on ``PYTHONPATH`` to import it. Intended full path would be
``/api/v1/crawl/...``.

Auth: none implemented (no ``x-user-id`` handling). Jobs are kept in a module-level dict,
so they are lost on restart and not shared between workers.
"""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from shared.logger import get_logger
from services.web_crawler import WebCrawler

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/crawl", tags=["crawl"])

# Global web crawler instance
# Single shared crawler. WebCrawler keeps per-crawl state on the instance (visited URLs,
# results), so overlapping crawls would corrupt each other.
web_crawler = WebCrawler()

# Store crawl jobs
# In-memory job registry keyed by job id (UUID string).
crawl_jobs: Dict[str, Dict[str, Any]] = {}


class CrawlRequest(BaseModel):
    """Request model for web crawling."""
    start_url: str = Field(..., description="Starting URL to crawl")
    max_depth: Optional[int] = Field(2, ge=0, le=5, description="Maximum crawl depth")
    max_pages: Optional[int] = Field(50, ge=1, le=500, description="Maximum pages to crawl")
    follow_links: Optional[bool] = Field(True, description="Whether to follow links")
    same_domain_only: Optional[bool] = Field(True, description="Only follow same-domain links")


class PageData(BaseModel):
    """Page data model."""
    url: str
    title: str
    description: Optional[str] = None
    content: str
    status_code: int
    depth: int


class CrawlResponse(BaseModel):
    """Response model for crawling results."""
    success: bool
    start_url: str
    pages_crawled: int
    pages: List[PageData]
    errors: List[Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CrawlStatusResponse(BaseModel):
    """Response model for crawl job status."""
    job_id: str
    status: str
    pages_crawled: int
    total_visited: int
    errors_count: int
    timestamp: datetime


@router.post("/start", response_model=CrawlResponse)
async def start_crawl(request: CrawlRequest):
    """POST /crawl/start - crawl synchronously and return all pages.

    Auth: none. Body: ``CrawlRequest``. Returns: ``CrawlResponse`` (``success`` is false when the
    crawler reported failure). Holds the request open until the crawl finishes.
    Errors: 400 (missing or non-http(s) URL; surfaces as 500, see below), 500 on failure.

    FIXME: the broad ``except Exception`` converts the 400s raised in the body into 500s.
    """
    try:
        if not request.start_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="start_url is required"
            )
        
        if not request.start_url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="URL must start with http:// or https://"
            )
        
        # Perform crawling
        result = await web_crawler.crawl(
            start_url=request.start_url,
            max_depth=request.max_depth or 2,
            max_pages=request.max_pages or 50,
            follow_links=request.follow_links if request.follow_links is not None else True,
            same_domain_only=request.same_domain_only if request.same_domain_only is not None else True
        )
        
        # Convert results
        pages = [
            PageData(
                url=page["url"],
                title=page.get("title", ""),
                description=page.get("description", ""),
                content=page.get("content", ""),
                status_code=page.get("status_code", 0),
                depth=page.get("depth", 0)
            )
            for page in result.get("pages", [])
        ]
        
        return CrawlResponse(
            success=result.get("status") == "completed",
            start_url=request.start_url,
            pages_crawled=result.get("pages_crawled", 0),
            pages=pages,
            errors=result.get("errors", [])
        )
    
    except Exception as e:
        logger.error(f"Web crawling failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Web crawling failed: {str(e)}"
        )


@router.post("/start-async")
async def start_crawl_async(
    request: CrawlRequest,
    background_tasks: BackgroundTasks
):
    """POST /crawl/start-async - start a crawl as a FastAPI background task.

    Auth: none. Body: ``CrawlRequest``. Returns: ``{"success", "job_id", "message", "status_url"}``;
    poll ``/crawl/status/{job_id}`` and fetch ``/crawl/results/{job_id}``.
    Errors: 400 (invalid URL; surfaces as 500 because of the broad ``except``), 500 on failure.
    """
    try:
        if not request.start_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="start_url is required"
            )
        
        if not request.start_url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="URL must start with http:// or https://"
            )
        
        # Imported lazily inside the handler (style quirk, not required).
        import uuid
        job_id = str(uuid.uuid4())
        
        # Store job info
        crawl_jobs[job_id] = {
            "status": "pending",
            "start_url": request.start_url,
            "pages_crawled": 0,
            "total_visited": 0,
            "errors": [],
            "created_at": datetime.utcnow()
        }
        
        # Add background task
        # Closure over ``job_id`` and ``request``; runs after the response is sent and records its outcome in ``crawl_jobs``.
        async def run_crawl():
            try:
                crawl_jobs[job_id]["status"] = "running"
                
                result = await web_crawler.crawl(
                    start_url=request.start_url,
                    max_depth=request.max_depth or 2,
                    max_pages=request.max_pages or 50,
                    follow_links=request.follow_links if request.follow_links is not None else True,
                    same_domain_only=request.same_domain_only if request.same_domain_only is not None else True
                )
                
                crawl_jobs[job_id].update({
                    "status": result.get("status", "completed"),
                    "pages_crawled": result.get("pages_crawled", 0),
                    "total_visited": result.get("total_visited", 0),
                    "errors": result.get("errors", []),
                    "pages": result.get("pages", []),
                    "completed_at": datetime.utcnow()
                })
            
            except Exception as e:
                logger.error(f"Background crawl failed: {str(e)}")
                crawl_jobs[job_id]["status"] = "failed"
                crawl_jobs[job_id]["error"] = str(e)
        
        background_tasks.add_task(run_crawl)
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Crawl started in background",
            "status_url": f"/crawl/status/{job_id}"
        }
    
    except Exception as e:
        logger.error(f"Failed to start async crawl: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start async crawl: {str(e)}"
        )


@router.get("/status/{job_id}", response_model=CrawlStatusResponse)
async def get_crawl_status(job_id: str):
    """GET /crawl/status/{job_id} - status and counters of a background crawl.

    Auth: none. Returns: ``CrawlStatusResponse``.
    Errors: 404 for an unknown job (but the broad ``except`` turns it into 500).

    FIXME: ``CrawlStatusResponse.timestamp`` is required but never passed here, so building the
    response raises a validation error and the endpoint currently always fails with 500.
    """
    try:
        if job_id not in crawl_jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job = crawl_jobs[job_id]
        
        return CrawlStatusResponse(
            job_id=job_id,
            status=job.get("status", "unknown"),
            pages_crawled=job.get("pages_crawled", 0),
            total_visited=job.get("total_visited", 0),
            errors_count=len(job.get("errors", []))
        )
    
    except Exception as e:
        logger.error(f"Failed to get crawl status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get crawl status: {str(e)}"
        )


@router.get("/results/{job_id}")
async def get_crawl_results(job_id: str):
    """GET /crawl/results/{job_id} - full results of a finished crawl.

    Auth: none. Returns: job id, status, start URL, counters, pages and errors.
    Errors: 404 unknown job, 202 while pending/running - both are re-wrapped as 500 by the broad
    ``except``.
    """
    try:
        if job_id not in crawl_jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job = crawl_jobs[job_id]
        
        if job.get("status") == "pending" or job.get("status") == "running":
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail="Crawl is still in progress"
            )
        
        return {
            "success": True,
            "job_id": job_id,
            "status": job.get("status"),
            "start_url": job.get("start_url"),
            "pages_crawled": job.get("pages_crawled", 0),
            "total_visited": job.get("total_visited", 0),
            "pages": job.get("pages", []),
            "errors": job.get("errors", [])
        }
    
    except Exception as e:
        logger.error(f"Failed to get crawl results: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get crawl results: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """GET /crawl/health - health probe with the number of pending/running jobs.

    Auth: none. Errors: 503 if the check fails.
    """
    try:
        return {
            "status": "healthy",
            "service": "web-crawler-service",
            "timestamp": datetime.utcnow().isoformat(),
            "active_jobs": len([j for j in crawl_jobs.values() if j.get("status") in ["pending", "running"]])
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Web crawler service is unhealthy"
        )
