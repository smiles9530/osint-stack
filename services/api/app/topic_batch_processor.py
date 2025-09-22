"""
Topic Discovery Batch Processor
Handles hourly batch processing for topic discovery and campaign detection
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

from .topic_discovery import topic_discovery_service
from .config import settings

logger = logging.getLogger("osint_api")

class TopicBatchProcessor:
    """Batch processor for topic discovery and campaign detection"""
    
    def __init__(self):
        self.is_running = False
        self.processing_interval = 3600  # 1 hour in seconds
        self.last_processing_time = None
    
    async def start_batch_processing(self):
        """Start the batch processing loop"""
        if self.is_running:
            logger.warning("Batch processing already running")
            return
        
        self.is_running = True
        logger.info("Starting topic discovery batch processing")
        
        try:
            while self.is_running:
                try:
                    await self._process_batch()
                    self.last_processing_time = datetime.now()
                    logger.info(f"Batch processing completed at {self.last_processing_time}")
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                
                # Wait for next processing cycle
                await asyncio.sleep(self.processing_interval)
                
        except asyncio.CancelledError:
            logger.info("Batch processing cancelled")
        finally:
            self.is_running = False
    
    async def stop_batch_processing(self):
        """Stop the batch processing loop"""
        self.is_running = False
        logger.info("Stopping topic discovery batch processing")
    
    async def _process_batch(self):
        """Process a single batch of topic discovery"""
        try:
            # Discover topics from last 24 hours
            logger.info("Starting topic discovery batch...")
            
            discovery_result = await topic_discovery_service.discover_topics(
                hours_back=24,
                min_documents=5,  # Lower threshold for batch processing
                similarity_threshold=0.7
            )
            
            if discovery_result.get('topics_discovered', 0) > 0:
                logger.info(f"Discovered {discovery_result['topics_discovered']} topics")
                
                # Detect campaigns
                campaigns = await topic_discovery_service.detect_campaigns(
                    min_volume_increase=2.0,
                    time_window_hours=6
                )
                
                if campaigns:
                    logger.warning(f"Detected {len(campaigns)} potential campaigns")
                    await self._handle_campaign_detection(campaigns)
                
                # Update topic trends
                await self._update_topic_trends()
                
            else:
                logger.info("No new topics discovered in this batch")
                
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
    
    async def _handle_campaign_detection(self, campaigns: List[Dict[str, Any]]):
        """Handle detected campaigns"""
        try:
            for campaign in campaigns:
                logger.warning(f"Campaign detected: {campaign['topic_name']} "
                             f"(Volume increase: {campaign['volume_increase']:.2f}x)")
                
                # Store campaign alert
                await self._store_campaign_alert(campaign)
                
                # Send notification (if configured)
                await self._send_campaign_notification(campaign)
                
        except Exception as e:
            logger.error(f"Campaign handling failed: {e}")
    
    async def _store_campaign_alert(self, campaign: Dict[str, Any]):
        """Store campaign alert in database"""
        try:
            from .db import get_conn
            
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO real_time_alerts 
                        (alert_type, severity, title, description, metadata, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        'campaign_detected',
                        campaign.get('severity', 'medium'),
                        f"Campaign Detected: {campaign['topic_name']}",
                        f"Topic volume increased by {campaign['volume_increase']:.2f}x "
                        f"({campaign['recent_volume']} articles in recent window)",
                        json.dumps(campaign),
                        campaign['detected_at']
                    ))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to store campaign alert: {e}")
    
    async def _send_campaign_notification(self, campaign: Dict[str, Any]):
        """Send campaign notification (placeholder for future implementation)"""
        try:
            # This could integrate with email, Slack, webhooks, etc.
            logger.info(f"Would send notification for campaign: {campaign['topic_name']}")
            
            # Example: Send to webhook
            # await self._send_webhook_notification(campaign)
            
        except Exception as e:
            logger.error(f"Campaign notification failed: {e}")
    
    async def _update_topic_trends(self):
        """Update topic trends and statistics"""
        try:
            # This could trigger additional analysis or caching
            logger.info("Updating topic trends...")
            
            # Get recent trends
            trends = await topic_discovery_service.get_topic_trends(days_back=7)
            
            # Store trend data for dashboard
            await self._store_trend_data(trends)
            
        except Exception as e:
            logger.error(f"Trend update failed: {e}")
    
    async def _store_trend_data(self, trends: Dict[str, Any]):
        """Store trend data for dashboard consumption"""
        try:
            from .db import get_conn
            
            with get_conn() as conn:
                with conn.cursor() as cur:
                    # Store trend summary
                    cur.execute("""
                        INSERT INTO analytics_cache 
                        (cache_key, cache_data, expires_at, created_at)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (cache_key) 
                        DO UPDATE SET
                            cache_data = EXCLUDED.cache_data,
                            expires_at = EXCLUDED.expires_at,
                            updated_at = NOW()
                    """, (
                        'topic_trends_7d',
                        json.dumps(trends),
                        datetime.now() + timedelta(hours=1),
                        datetime.now()
                    ))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to store trend data: {e}")
    
    async def process_immediate(self, hours_back: int = 24) -> Dict[str, Any]:
        """Process topic discovery immediately (for testing/manual triggers)"""
        try:
            logger.info(f"Processing immediate topic discovery for last {hours_back} hours")
            
            result = await topic_discovery_service.discover_topics(
                hours_back=hours_back,
                min_documents=5,
                similarity_threshold=0.7
            )
            
            # Also detect campaigns
            campaigns = await topic_discovery_service.detect_campaigns(
                min_volume_increase=2.0,
                time_window_hours=6
            )
            
            result['campaigns_detected'] = len(campaigns)
            result['campaigns'] = campaigns
            
            return result
            
        except Exception as e:
            logger.error(f"Immediate processing failed: {e}")
            return {
                'error': str(e),
                'topics': [],
                'assignments': [],
                'campaigns': []
            }
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        return {
            'is_running': self.is_running,
            'last_processing_time': self.last_processing_time.isoformat() if self.last_processing_time else None,
            'processing_interval_seconds': self.processing_interval,
            'next_processing_in_seconds': self._get_next_processing_time()
        }
    
    def _get_next_processing_time(self) -> Optional[int]:
        """Get seconds until next processing"""
        if not self.last_processing_time:
            return None
        
        next_time = self.last_processing_time + timedelta(seconds=self.processing_interval)
        now = datetime.now()
        
        if next_time > now:
            return int((next_time - now).total_seconds())
        else:
            return 0

# Global instance
topic_batch_processor = TopicBatchProcessor()


