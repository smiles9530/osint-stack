"""
Puppeteer Service for web scraping and automation
Uses Playwright instead of pyppeteer for better compatibility
"""
import asyncio
import base64
import io
from typing import Dict, List, Optional, Any
from playwright.async_api import async_playwright, Browser, Page
from .logging_config import logger
from .enhanced_error_handling import APIError, ErrorHandler

class PuppeteerService:
    """Service for web scraping and automation using Playwright"""
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.playwright = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize Playwright browser"""
        try:
            if not self.is_initialized:
                logger.info("Initializing Playwright browser...")
                self.playwright = await async_playwright().start()
                self.browser = await self.playwright.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-accelerated-2d-canvas',
                        '--no-first-run',
                        '--no-zygote',
                        '--disable-gpu',
                        '--disable-web-security',
                        '--disable-features=VizDisplayCompositor'
                    ]
                )
                self.is_initialized = True
                logger.info("Playwright browser initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Playwright: {e}")
            raise ServiceUnavailableError("Playwright browser initialization failed")
    
    async def close(self):
        """Close Playwright browser"""
        if self.browser:
            await self.browser.close()
            self.browser = None
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None
        self.is_initialized = False
        logger.info("Playwright browser closed")
    
    async def get_page(self) -> Page:
        """Get a new page from the browser"""
        if not self.is_initialized:
            await self.initialize()
        
        if not self.browser:
            raise ServiceUnavailableError("Browser not initialized")
        
        return await self.browser.new_page()
    
    async def scrape_url(self, url: str, wait_for: str = "networkidle", timeout: int = 30000, 
                        user_agent: str = "OSINT-Stack/1.0") -> Dict[str, Any]:
        """Scrape content from a URL"""
        page = None
        try:
            page = await self.get_page()
            
            # Set user agent
            await page.set_extra_http_headers({"User-Agent": user_agent})
            
            # Navigate to URL
            await page.goto(url, wait_until=wait_for, timeout=timeout)
            
            # Get page content
            content = await page.content()
            title = await page.title()
            
            # Get metadata
            metadata = {
                "url": url,
                "title": title,
                "timestamp": asyncio.get_event_loop().time(),
                "user_agent": user_agent
            }
            
            return {
                "success": True,
                "content": content,
                "metadata": metadata,
                "can_scrape": True
            }
            
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "can_scrape": False
            }
        finally:
            if page:
                await page.close()
    
    async def take_screenshot(self, url: str, full_page: bool = True, format: str = "png") -> Dict[str, Any]:
        """Take a screenshot of a URL"""
        page = None
        try:
            page = await self.get_page()
            await page.goto(url, wait_until="networkidle")
            
            screenshot_bytes = await page.screenshot(full_page=full_page, type=format)
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            return {
                "success": True,
                "screenshot": f"data:image/{format};base64,{screenshot_b64}",
                "url": url,
                "full_page": full_page,
                "format": format
            }
            
        except Exception as e:
            logger.error(f"Error taking screenshot of {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
        finally:
            if page:
                await page.close()
    
    async def generate_pdf(self, url: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate PDF from URL"""
        page = None
        try:
            page = await self.get_page()
            await page.goto(url, wait_until="networkidle")
            
            pdf_options = {
                "format": "A4",
                "print_background": True,
                "margin": {"top": "1cm", "right": "1cm", "bottom": "1cm", "left": "1cm"}
            }
            if options:
                pdf_options.update(options)
            
            pdf_bytes = await page.pdf(**pdf_options)
            pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')
            
            return {
                "success": True,
                "pdf": f"data:application/pdf;base64,{pdf_b64}",
                "url": url,
                "options": pdf_options
            }
            
        except Exception as e:
            logger.error(f"Error generating PDF from {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
        finally:
            if page:
                await page.close()
    
    async def get_page_metadata(self, url: str) -> Dict[str, Any]:
        """Get page metadata"""
        page = None
        try:
            page = await self.get_page()
            await page.goto(url, wait_until="networkidle")
            
            # Extract metadata
            title = await page.title()
            description = await page.get_attribute('meta[name="description"]', 'content') or ""
            keywords = await page.get_attribute('meta[name="keywords"]', 'content') or ""
            author = await page.get_attribute('meta[name="author"]', 'content') or ""
            
            # Get all meta tags
            meta_tags = await page.evaluate("""
                () => {
                    const metas = Array.from(document.querySelectorAll('meta'));
                    return metas.map(meta => ({
                        name: meta.getAttribute('name'),
                        property: meta.getAttribute('property'),
                        content: meta.getAttribute('content')
                    })).filter(meta => meta.content);
                }
            """)
            
            return {
                "success": True,
                "url": url,
                "title": title,
                "description": description,
                "keywords": keywords,
                "author": author,
                "meta_tags": meta_tags
            }
            
        except Exception as e:
            logger.error(f"Error getting metadata from {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
        finally:
            if page:
                await page.close()
    
    async def scrape_multiple_urls(self, urls: List[str], max_concurrent: int = 5) -> Dict[str, Any]:
        """Scrape multiple URLs concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_single(url):
            async with semaphore:
                return await self.scrape_url(url)
        
        try:
            tasks = [scrape_single(url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful = []
            failed = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed.append({
                        "url": urls[i],
                        "error": str(result)
                    })
                elif result.get("success"):
                    successful.append(result)
                else:
                    failed.append({
                        "url": urls[i],
                        "error": result.get("error", "Unknown error")
                    })
            
            return {
                "success": True,
                "total": len(urls),
                "successful": len(successful),
                "failed": len(failed),
                "results": successful,
                "errors": failed
            }
            
        except Exception as e:
            logger.error(f"Error scraping multiple URLs: {e}")
            return {
                "success": False,
                "error": str(e),
                "total": len(urls),
                "successful": 0,
                "failed": len(urls)
            }

# Global instance
puppeteer_service = PuppeteerService()