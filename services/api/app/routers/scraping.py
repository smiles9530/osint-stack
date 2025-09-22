"""
Web Scraping router
Handles Puppeteer scraping and web automation
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, Dict, Any, List
from datetime import datetime
from ..auth import get_current_active_user, User
from ..schemas import (
    PuppeteerScrapeRequest, PuppeteerScreenshotRequest,
    PuppeteerPdfRequest, PuppeteerMetadataRequest, PuppeteerScrapeMultipleRequest,
    PuppeteerScrapeFlexibleRequest, PuppeteerResponse, ErrorResponse
)
from ..puppeteer_service import puppeteer_service

router = APIRouter(prefix="/scraping", tags=["Web Scraping"])

# Puppeteer endpoints
@router.post("/puppeteer/scrape")
async def scrape_url_endpoint(
    request: PuppeteerScrapeRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Scrape URL using Puppeteer"""
    try:
        result = await puppeteer_service.scrape_url(
            request.url,
            request.wait_for,
            request.timeout,
            request.user_agent
        )
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"URL scraping failed: {str(e)}"
        )

@router.post("/puppeteer/screenshot")
async def take_screenshot_endpoint(
    request: PuppeteerScreenshotRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Take screenshot of URL using Puppeteer"""
    try:
        result = await puppeteer_service.take_screenshot(
            request.url,
            request.full_page,
            request.format
        )
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Screenshot failed: {str(e)}"
        )

@router.post("/puppeteer/pdf")
async def generate_pdf_endpoint(
    request: PuppeteerPdfRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Generate PDF from URL using Puppeteer"""
    try:
        result = await puppeteer_service.generate_pdf(
            request.url,
            request.options
        )
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation failed: {str(e)}"
        )

@router.post("/puppeteer/metadata")
async def get_page_metadata_endpoint(
    request: PuppeteerMetadataRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Get page metadata using Puppeteer"""
    try:
        result = await puppeteer_service.get_page_metadata(request.url)
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Metadata extraction failed: {str(e)}"
        )

@router.post("/puppeteer/scrape-multiple")
async def scrape_multiple_urls_endpoint(
    request: PuppeteerScrapeMultipleRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Scrape multiple URLs using Puppeteer"""
    try:
        result = await puppeteer_service.scrape_multiple_urls(
            request.urls,
            request.max_concurrent
        )
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Multiple URL scraping failed: {str(e)}"
        )

@router.post("/puppeteer/scrape-flexible")
async def scrape_flexible_urls_endpoint(
    request: PuppeteerScrapeFlexibleRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Scrape URLs using Puppeteer - accepts both single URLs and arrays"""
    try:
        # The validator in PuppeteerScrapeFlexibleRequest already converts single URLs to lists
        result = await puppeteer_service.scrape_multiple_urls(
            request.urls,  # This is now guaranteed to be a list
            request.max_concurrent
        )
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Flexible URL scraping failed: {str(e)}"
        )