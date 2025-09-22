"""
Enhanced Feedback router
Handles advanced user feedback submission, ML-powered digest generation, and recommendation system
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from fastapi import APIRouter, Depends, Query, HTTPException, BackgroundTasks
import numpy as np
import pandas as pd
# Optional sklearn imports for ML features
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Create dummy classes for when sklearn is not available
    class TfidfVectorizer:
        def __init__(self, *args, **kwargs):
            pass
    class KMeans:
        def __init__(self, *args, **kwargs):
            pass
    class LatentDirichletAllocation:
        def __init__(self, *args, **kwargs):
            pass
    def cosine_similarity(*args, **kwargs):
        return [[0.0]]

from ..auth import get_current_active_user, User
from ..schemas import FeedbackSubmission, FeedbackResponse, Digest, DigestList, ErrorResponse
from ..enhanced_error_handling import ErrorHandler, APIError
from ..cache import cache
from ..monitoring import monitoring_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feedback", tags=["Feedback"])

# Advanced feedback analysis service
class AdvancedFeedbackService:
    def __init__(self):
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.lda_model = None
        else:
            self.vectorizer = None
            self.lda_model = None
        self.user_preferences = {}
        self.content_recommendations = {}
    
    async def analyze_feedback_sentiment(self, feedback_text: str) -> Dict[str, Any]:
        """Analyze sentiment of feedback text using ML pipeline"""
        try:
            from ..ml_pipeline import ml_pipeline
            
            if not feedback_text or len(feedback_text.strip()) < 3:
                return {"sentiment": "neutral", "confidence": 0.0, "emotions": []}
            
            # Analyze sentiment
            sentiment_result = await ml_pipeline.analyze_sentiment(feedback_text)
            
            # Extract emotions (simplified)
            emotions = []
            if sentiment_result.get("scores", {}).get("negative", 0) > 0.7:
                emotions.append("frustrated")
            if sentiment_result.get("scores", {}).get("positive", 0) > 0.7:
                emotions.append("satisfied")
            if "helpful" in feedback_text.lower():
                emotions.append("helpful")
            if "confusing" in feedback_text.lower():
                emotions.append("confused")
            
            return {
                "sentiment": sentiment_result.get("sentiment", "neutral"),
                "confidence": sentiment_result.get("confidence", 0.0),
                "emotions": emotions,
                "scores": sentiment_result.get("scores", {})
            }
            
        except Exception as e:
            logger.warning(f"Feedback sentiment analysis failed: {e}")
            return {"sentiment": "neutral", "confidence": 0.0, "emotions": []}
    
    async def generate_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """Generate user preferences based on feedback history"""
        try:
            from .. import db
            
            with db.get_conn() as conn:
                with conn.cursor() as cur:
                    # Get user's feedback history
                    cur.execute("""
                        SELECT 
                            uf.feedback_score,
                            uf.feedback_text,
                            uf.clicked,
                            uf.upvote,
                            a.title,
                            a.content,
                            aa.sentiment_distribution,
                            aa.bias_scores,
                            aa.topic
                        FROM user_feedback uf
                        JOIN articles a ON uf.article_id = a.id
                        LEFT JOIN aggregated_analysis aa ON a.id = aa.article_id
                        WHERE uf.user_id = %s
                        AND uf.submitted_at >= %s
                        ORDER BY uf.submitted_at DESC
                        LIMIT 100
                    """, (user_id, datetime.utcnow() - timedelta(days=30)))
                    
                    feedback_data = cur.fetchall()
            
            if not feedback_data:
                return {"preferences": {}, "confidence": 0.0}
            
            # Analyze preferences
            preferences = {
                "preferred_topics": {},
                "sentiment_preferences": {},
                "content_preferences": {},
                "engagement_patterns": {}
            }
            
            # Topic preferences
            topic_scores = {}
            for row in feedback_data:
                topic = row[8] or "general"
                score = row[0] or 0.5
                if topic not in topic_scores:
                    topic_scores[topic] = []
                topic_scores[topic].append(score)
            
            for topic, scores in topic_scores.items():
                preferences["preferred_topics"][topic] = {
                    "avg_score": np.mean(scores),
                    "interaction_count": len(scores),
                    "preference_strength": np.std(scores) < 0.3  # Low variance = strong preference
                }
            
            # Sentiment preferences
            sentiment_scores = {"positive": [], "negative": [], "neutral": []}
            for row in feedback_data:
                if row[6]:  # sentiment_distribution
                    sentiment_data = row[6] if isinstance(row[6], dict) else json.loads(row[6])
                    user_score = row[0] or 0.5
                    
                    for sentiment, score in sentiment_data.items():
                        if sentiment in sentiment_scores:
                            sentiment_scores[sentiment].append((score, user_score))
            
            for sentiment, score_pairs in sentiment_scores.items():
                if score_pairs:
                    content_scores = [pair[0] for pair in score_pairs]
                    user_scores = [pair[1] for pair in score_pairs]
                    correlation = np.corrcoef(content_scores, user_scores)[0, 1] if len(content_scores) > 1 else 0
                    preferences["sentiment_preferences"][sentiment] = {
                        "correlation": correlation,
                        "avg_user_score": np.mean(user_scores),
                        "sample_size": len(score_pairs)
                    }
            
            # Content preferences
            clicked_articles = [row for row in feedback_data if row[2]]  # clicked = True
            upvoted_articles = [row for row in feedback_data if row[3]]  # upvote = True
            
            preferences["content_preferences"] = {
                "click_rate": len(clicked_articles) / len(feedback_data) if feedback_data else 0,
                "upvote_rate": len(upvoted_articles) / len(feedback_data) if feedback_data else 0,
                "avg_feedback_score": np.mean([row[0] for row in feedback_data if row[0] is not None]) if any(row[0] for row in feedback_data) else 0
            }
            
            # Engagement patterns
            preferences["engagement_patterns"] = {
                "total_interactions": len(feedback_data),
                "text_feedback_rate": len([row for row in feedback_data if row[1]]) / len(feedback_data) if feedback_data else 0,
                "high_score_rate": len([row for row in feedback_data if (row[0] or 0) > 0.7]) / len(feedback_data) if feedback_data else 0
            }
            
            # Calculate overall confidence
            confidence = min(1.0, len(feedback_data) / 50)  # More data = higher confidence
            
            return {
                "preferences": preferences,
                "confidence": confidence,
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate user preferences: {e}")
            return {"preferences": {}, "confidence": 0.0}
    
    async def generate_content_recommendations(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Generate personalized content recommendations based on user preferences"""
        try:
            from .. import db
            
            # Get user preferences
            user_prefs = await self.generate_user_preferences(user_id)
            if user_prefs["confidence"] < 0.3:
                return []  # Not enough data for recommendations
            
            preferences = user_prefs["preferences"]
            
            with db.get_conn() as conn:
                with conn.cursor() as cur:
                    # Get recent articles with analysis
                    cur.execute("""
                        SELECT 
                            a.id, a.title, a.content, a.url, a.published_at,
                            aa.sentiment_distribution, aa.bias_scores, aa.topic,
                            aa.confidence_avg, aa.risk_score
                        FROM articles a
                        LEFT JOIN aggregated_analysis aa ON a.id = aa.article_id
                        WHERE a.published_at >= %s
                        AND a.is_deleted = FALSE
                        ORDER BY a.published_at DESC
                        LIMIT 200
                    """, (datetime.utcnow() - timedelta(days=7),))
                    
                    articles = cur.fetchall()
            
            if not articles:
                return []
            
            # Score articles based on user preferences
            scored_articles = []
            
            for article in articles:
                score = 0.0
                reasons = []
                
                # Topic preference scoring
                article_topic = article[7] or "general"
                if article_topic in preferences["preferred_topics"]:
                    topic_pref = preferences["preferred_topics"][article_topic]
                    score += topic_pref["avg_score"] * 0.4
                    reasons.append(f"Matches preferred topic: {article_topic}")
                
                # Sentiment preference scoring
                if article[5]:  # sentiment_distribution
                    sentiment_data = article[5] if isinstance(article[5], dict) else json.loads(article[5])
                    for sentiment, content_score in sentiment_data.items():
                        if sentiment in preferences["sentiment_preferences"]:
                            sentiment_pref = preferences["sentiment_preferences"][sentiment]
                            if sentiment_pref["correlation"] > 0.3:  # Positive correlation
                                score += content_score * sentiment_pref["correlation"] * 0.3
                                reasons.append(f"Matches {sentiment} sentiment preference")
                
                # Content quality scoring
                confidence = article[8] or 0.0
                risk_score = article[9] or 0.0
                score += confidence * 0.2  # Higher confidence = better
                score -= risk_score * 0.1  # Lower risk = better
                
                # Recency bonus
                days_old = (datetime.utcnow() - article[4]).days
                recency_bonus = max(0, 1 - (days_old / 7))  # Decay over 7 days
                score += recency_bonus * 0.1
                
                scored_articles.append({
                    "article_id": str(article[0]),
                    "title": article[1],
                    "content": article[2][:200] + "..." if len(article[2]) > 200 else article[2],
                    "url": article[3],
                    "published_at": article[4].isoformat(),
                    "topic": article_topic,
                    "confidence": confidence,
                    "risk_score": risk_score,
                    "recommendation_score": score,
                    "reasons": reasons
                })
            
            # Sort by recommendation score and return top results
            scored_articles.sort(key=lambda x: x["recommendation_score"], reverse=True)
            
            return scored_articles[:limit]
            
        except Exception as e:
            logger.error(f"Failed to generate content recommendations: {e}")
            return []
    
    async def generate_ml_digest(self, topic: Optional[str] = None, days_back: int = 7, 
                               user_id: Optional[int] = None) -> Dict[str, Any]:
        """Generate ML-powered digest with advanced content analysis"""
        try:
            from .. import db
            from ..ml_pipeline import ml_pipeline
            
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            with db.get_conn() as conn:
                with conn.cursor() as cur:
                    # Get articles with comprehensive analysis
                    query = """
                        SELECT 
                            a.id, a.title, a.content, a.url, a.published_at,
                            aa.sentiment_distribution, aa.bias_scores, aa.confidence_avg,
                            aa.topic, aa.risk_score, aa.analysis_data
                        FROM articles a
                        LEFT JOIN aggregated_analysis aa ON a.id = aa.article_id
                        WHERE a.published_at >= %s AND a.published_at <= %s
                        AND a.is_deleted = FALSE
                    """
                    params = [start_date, end_date]
                    
                    if topic:
                        query += " AND aa.topic = %s"
                        params.append(topic)
                    
                    query += " ORDER BY a.published_at DESC LIMIT 200"
                    
                    cur.execute(query, params)
                    articles = cur.fetchall()
                    
                    if not articles:
                        raise APIError(
                            status_code=404,
                            error_code="NO_ARTICLES_FOUND",
                            message="No articles found for digest generation"
                        )
                    
                    # Get user feedback for engagement analysis
                    article_ids = [str(row[0]) for row in articles]
                    cur.execute("""
                        SELECT 
                            article_id,
                            AVG(feedback_score) as avg_score,
                            COUNT(*) as feedback_count,
                            SUM(CASE WHEN upvote = TRUE THEN 1 ELSE 0 END) as upvotes,
                            SUM(CASE WHEN clicked = TRUE THEN 1 ELSE 0 END) as clicks,
                            COUNT(CASE WHEN feedback_text IS NOT NULL THEN 1 END) as text_feedback_count
                        FROM user_feedback
                        WHERE article_id = ANY(%s)
                        GROUP BY article_id
                    """, (article_ids,))
                    
                    feedback_data = {row[0]: {
                        "avg_score": float(row[1]) if row[1] else 0.0,
                        "feedback_count": row[2],
                        "upvotes": row[3],
                        "clicks": row[4],
                        "text_feedback_count": row[5]
                    } for row in cur.fetchall()}
            
            # Advanced content analysis
            article_texts = [row[2] for row in articles if row[2]]
            
            # Extract topics using LDA
            topics = await self._extract_topics_lda(article_texts[:50])  # Limit for performance
            
            # Sentiment analysis
            sentiment_analysis = await self._analyze_sentiment_trends(articles)
            
            # Content clustering
            content_clusters = await self._cluster_content(article_texts[:30])
            
            # Generate insights
            insights = await self._generate_content_insights(articles, feedback_data, topics)
            
            # Create comprehensive digest
            digest_id = str(uuid.uuid4())
            digest_title = f"Intelligence Digest - {topic or 'General News'}"
            
            # Generate digest content
            digest_content = await self._generate_digest_content(
                articles, feedback_data, topics, sentiment_analysis, 
                content_clusters, insights, topic, days_back
            )
            
            # Store digest
            with db.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO digests 
                        (id, title, content, topic, created_at, article_count, 
                         sentiment_summary, key_insights, analysis_metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        digest_id,
                        digest_title,
                        digest_content,
                        topic or "general",
                        datetime.utcnow(),
                        len(articles),
                        json.dumps(sentiment_analysis),
                        json.dumps(insights),
                        json.dumps({
                            "topics": topics,
                            "clusters": content_clusters,
                            "generation_method": "ml_enhanced",
                            "user_id": user_id
                        })
                    ))
            
            return {
                "digest_id": digest_id,
                "title": digest_title,
                "article_count": len(articles),
                "topic": topic or "general",
                "insights": insights,
                "sentiment_analysis": sentiment_analysis,
                "topics": topics,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except APIError:
            raise
        except Exception as e:
            logger.error(f"ML digest generation failed: {e}")
            raise APIError(
                status_code=500,
                error_code="DIGEST_GENERATION_FAILED",
                message="ML digest generation failed",
                detail=str(e)
            )
    
    async def _extract_topics_lda(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract topics using Latent Dirichlet Allocation"""
        try:
            if not SKLEARN_AVAILABLE or not texts or len(texts) < 3:
                return []
            
            # Prepare texts
            processed_texts = [text[:1000] for text in texts if text and len(text.strip()) > 10]
            
            if len(processed_texts) < 3:
                return []
            
            # Vectorize texts
            tf_matrix = self.vectorizer.fit_transform(processed_texts)
            
            # Apply LDA
            n_topics = min(5, len(processed_texts) // 3)
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(tf_matrix)
            
            # Extract topics
            feature_names = self.vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[-5:][::-1]]
                topics.append({
                    "id": topic_idx,
                    "name": " ".join(top_words[:3]),
                    "keywords": top_words,
                    "weight": float(np.mean(topic))
                })
            
            return topics
            
        except Exception as e:
            logger.warning(f"LDA topic extraction failed: {e}")
            return []
    
    async def _analyze_sentiment_trends(self, articles: List[Tuple]) -> Dict[str, Any]:
        """Analyze sentiment trends across articles"""
        try:
            sentiment_scores = {"positive": [], "negative": [], "neutral": []}
            daily_sentiments = {}
            
            for article in articles:
                if article[5]:  # sentiment_distribution
                    sentiment_data = article[5] if isinstance(article[5], dict) else json.loads(article[5])
                    date = article[4].date()
                    
                    if date not in daily_sentiments:
                        daily_sentiments[date] = {"positive": [], "negative": [], "neutral": []}
                    
                    for sentiment, score in sentiment_data.items():
                        if sentiment in sentiment_scores:
                            sentiment_scores[sentiment].append(score)
                            daily_sentiments[date][sentiment].append(score)
            
            # Calculate trends
            trends = {}
            for sentiment, scores in sentiment_scores.items():
                if scores:
                    trends[sentiment] = {
                        "average": float(np.mean(scores)),
                        "trend": "increasing" if len(scores) > 1 and scores[-1] > scores[0] else "decreasing",
                        "volatility": float(np.std(scores))
                    }
            
            # Daily trends
            daily_trends = []
            for date, sentiments in sorted(daily_sentiments.items()):
                daily_avg = {}
                for sentiment, scores in sentiments.items():
                    if scores:
                        daily_avg[sentiment] = float(np.mean(scores))
                
                if daily_avg:
                    daily_trends.append({
                        "date": date.isoformat(),
                        "sentiments": daily_avg
                    })
            
            return {
                "overall_trends": trends,
                "daily_trends": daily_trends,
                "analysis_period": len(articles)
            }
            
        except Exception as e:
            logger.warning(f"Sentiment trend analysis failed: {e}")
            return {"overall_trends": {}, "daily_trends": [], "analysis_period": 0}
    
    async def _cluster_content(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Cluster content into thematic groups"""
        try:
            if not SKLEARN_AVAILABLE or not texts or len(texts) < 3:
                return []
            
            # Vectorize texts
            tf_matrix = self.vectorizer.fit_transform(texts)
            
            # Cluster using K-means
            n_clusters = min(3, len(texts) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(tf_matrix)
            
            # Group texts by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    "text": texts[i][:200] + "..." if len(texts[i]) > 200 else texts[i],
                    "index": i
                })
            
            # Generate cluster summaries
            cluster_summaries = []
            for cluster_id, cluster_texts in clusters.items():
                # Find most representative text (closest to centroid)
                centroid = kmeans.cluster_centers_[cluster_id]
                similarities = cosine_similarity(tf_matrix[cluster_id], centroid.reshape(1, -1))
                most_representative_idx = np.argmax(similarities)
                
                cluster_summaries.append({
                    "cluster_id": int(cluster_id),
                    "size": len(cluster_texts),
                    "representative_text": cluster_texts[most_representative_idx]["text"],
                    "themes": self._extract_cluster_themes(cluster_texts)
                })
            
            return cluster_summaries
            
        except Exception as e:
            logger.warning(f"Content clustering failed: {e}")
            return []
    
    def _extract_cluster_themes(self, cluster_texts: List[Dict[str, Any]]) -> List[str]:
        """Extract themes from clustered texts"""
        try:
            # Simple keyword extraction
            all_text = " ".join([text["text"] for text in cluster_texts])
            words = all_text.lower().split()
            
            # Remove common words and get top keywords
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            keywords = [word for word in words if word not in stop_words and len(word) > 3]
            
            # Count word frequency
            word_freq = {}
            for word in keywords:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Return top themes
            themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            return [theme[0] for theme in themes]
            
        except Exception as e:
            logger.warning(f"Theme extraction failed: {e}")
            return []
    
    async def _generate_content_insights(self, articles: List[Tuple], feedback_data: Dict[str, Dict], 
                                       topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate content insights based on analysis"""
        try:
            insights = []
            
            # Engagement insights
            if feedback_data:
                avg_engagement = np.mean([data["avg_score"] for data in feedback_data.values() if data["avg_score"] > 0])
                high_engagement_articles = [aid for aid, data in feedback_data.items() if data["avg_score"] > avg_engagement + 0.2]
                
                if high_engagement_articles:
                    insights.append({
                        "type": "engagement",
                        "title": "High Engagement Content",
                        "description": f"{len(high_engagement_articles)} articles showing above-average engagement",
                        "value": len(high_engagement_articles),
                        "severity": "info"
                    })
            
            # Topic insights
            if topics:
                dominant_topic = max(topics, key=lambda t: t["weight"])
                insights.append({
                    "type": "topics",
                    "title": "Dominant Topic",
                    "description": f"Most discussed topic: {dominant_topic['name']}",
                    "value": dominant_topic["name"],
                    "severity": "info"
                })
            
            # Content quality insights
            high_confidence_articles = [article for article in articles if (article[7] or 0) > 0.8]
            if high_confidence_articles:
                insights.append({
                    "type": "quality",
                    "title": "High Quality Content",
                    "description": f"{len(high_confidence_articles)} articles with high confidence scores",
                    "value": len(high_confidence_articles),
                    "severity": "success"
                })
            
            # Risk insights
            high_risk_articles = [article for article in articles if (article[9] or 0) > 0.7]
            if high_risk_articles:
                insights.append({
                    "type": "risk",
                    "title": "High Risk Content Detected",
                    "description": f"{len(high_risk_articles)} articles flagged as high risk",
                    "value": len(high_risk_articles),
                    "severity": "warning"
                })
            
            return insights
            
        except Exception as e:
            logger.warning(f"Content insights generation failed: {e}")
            return []
    
    async def _generate_digest_content(self, articles: List[Tuple], feedback_data: Dict[str, Dict],
                                     topics: List[Dict[str, Any]], sentiment_analysis: Dict[str, Any],
                                     content_clusters: List[Dict[str, Any]], insights: List[Dict[str, Any]],
                                     topic: Optional[str], days_back: int) -> str:
        """Generate comprehensive digest content"""
        try:
            digest_title = f"Intelligence Digest - {topic or 'General News'}"
            
            # Build digest content
            content_parts = [
                f"# {digest_title}",
                f"",
                f"## Executive Summary",
                f"Generated from {len(articles)} articles over the past {days_back} days.",
                f"Analysis period: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                f""
            ]
            
            # Key topics section
            if topics:
                content_parts.extend([
                    f"## Key Topics",
                    f""
                ])
                for i, topic_info in enumerate(topics[:5], 1):
                    content_parts.append(f"{i}. **{topic_info['name']}** - {', '.join(topic_info['keywords'][:3])}")
                content_parts.append("")
            
            # Sentiment analysis section
            if sentiment_analysis.get("overall_trends"):
                content_parts.extend([
                    f"## Sentiment Analysis",
                    f""
                ])
                for sentiment, data in sentiment_analysis["overall_trends"].items():
                    trend_arrow = "📈" if data["trend"] == "increasing" else "📉" if data["trend"] == "decreasing" else "➡️"
                    content_parts.append(f"- **{sentiment.title()}**: {data['average']:.2f} {trend_arrow}")
                content_parts.append("")
            
            # Content clusters section
            if content_clusters:
                content_parts.extend([
                    f"## Content Themes",
                    f""
                ])
                for cluster in content_clusters[:3]:
                    content_parts.append(f"### Theme {cluster['cluster_id'] + 1} ({cluster['size']} articles)")
                    content_parts.append(f"**Representative**: {cluster['representative_text'][:100]}...")
                    if cluster['themes']:
                        content_parts.append(f"**Keywords**: {', '.join(cluster['themes'][:5])}")
                    content_parts.append("")
            
            # Top articles section
            content_parts.extend([
                f"## Top Articles by Engagement",
                f""
            ])
            
            # Sort articles by engagement
            article_engagement = []
            for article in articles:
                article_id = str(article[0])
                engagement_score = 0
                if article_id in feedback_data:
                    data = feedback_data[article_id]
                    engagement_score = (data.get("avg_score", 0) * 0.4 + 
                                      data.get("upvotes", 0) * 0.3 + 
                                      data.get("clicks", 0) * 0.3)
                
                article_engagement.append((article, engagement_score))
            
            article_engagement.sort(key=lambda x: x[1], reverse=True)
            
            for i, (article, score) in enumerate(article_engagement[:5], 1):
                content_parts.append(f"{i}. **{article[1]}** (Engagement: {score:.2f})")
                if article[2]:
                    content_parts.append(f"   {article[2][:150]}...")
                content_parts.append("")
            
            # Insights section
            if insights:
                content_parts.extend([
                    f"## Key Insights",
                    f""
                ])
                for insight in insights:
                    severity_icon = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(insight["severity"], "ℹ️")
                    content_parts.append(f"{severity_icon} **{insight['title']}**: {insight['description']}")
                content_parts.append("")
            
            # Footer
            content_parts.extend([
                f"---",
                f"*Generated by OSINT Stack Intelligence System*",
                f"*Digest ID: {str(uuid.uuid4())[:8]}*"
            ])
            
            return "\n".join(content_parts)
            
        except Exception as e:
            logger.error(f"Digest content generation failed: {e}")
            return f"# {digest_title}\n\nError generating digest content: {str(e)}"

feedback_service = AdvancedFeedbackService()

@router.post(
    "/submit",
    summary="Advanced Feedback Submission",
    description="Submit user feedback with ML-powered sentiment analysis and preference learning",
    responses={
        200: {"description": "Feedback submitted successfully"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        404: {"description": "Article not found", "model": ErrorResponse},
        500: {"description": "Feedback submission failed", "model": ErrorResponse}
    }
)
async def submit_feedback_advanced(
    article_id: int,
    clicked: Optional[bool] = None,
    upvote: Optional[bool] = None,
    correct_after_days: Optional[bool] = None,
    feedback_score: Optional[float] = None,
    feedback_text: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Advanced feedback submission with ML analysis"""
    try:
        from .. import db
        
        # Validate feedback score if provided
        if feedback_score is not None and not (0.0 <= feedback_score <= 1.0):
            ErrorHandler.raise_bad_request("Feedback score must be between 0.0 and 1.0")
        
        # Check if article exists
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM articles WHERE id = %s", (article_id,))
                if not cur.fetchone():
                    ErrorHandler.raise_not_found("Article not found")
        
        # Analyze feedback sentiment if text provided
        sentiment_analysis = {}
        if feedback_text and len(feedback_text.strip()) > 3:
            sentiment_analysis = await feedback_service.analyze_feedback_sentiment(feedback_text)
        
        # Store feedback with analysis
        feedback_id = str(uuid.uuid4())
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_feedback 
                    (id, article_id, user_id, clicked, upvote, correct_after_days, 
                     feedback_score, feedback_text, submitted_at, sentiment_analysis)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_id, user_id) 
                    DO UPDATE SET
                        clicked = EXCLUDED.clicked,
                        upvote = EXCLUDED.upvote,
                        correct_after_days = EXCLUDED.correct_after_days,
                        feedback_score = EXCLUDED.feedback_score,
                        feedback_text = EXCLUDED.feedback_text,
                        submitted_at = EXCLUDED.submitted_at,
                        sentiment_analysis = EXCLUDED.sentiment_analysis
                """, (
                    feedback_id,
                    article_id,
                    current_user.id,
                    clicked,
                    upvote,
                    correct_after_days,
                    feedback_score,
                    feedback_text,
                    datetime.utcnow(),
                    json.dumps(sentiment_analysis) if sentiment_analysis else None
                ))
        
        # Update user preferences cache
        await feedback_service._invalidate_user_preferences_cache(current_user.id)
        
        return {
            "message": "Feedback submitted successfully",
            "feedback_id": feedback_id,
            "article_id": article_id,
            "sentiment_analysis": sentiment_analysis,
            "submitted_at": datetime.utcnow().isoformat()
        }
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        ErrorHandler.raise_internal_server_error("Feedback submission failed", str(e))

@router.post(
    "/submit/batch",
    summary="Batch Feedback Submission",
    description="Submit multiple feedback entries in parallel",
    responses={
        200: {"description": "Batch feedback submitted successfully"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        500: {"description": "Batch feedback submission failed", "model": ErrorResponse}
    }
)
async def submit_feedback_batch(
    feedback_entries: List[Dict[str, Any]],
    current_user: User = Depends(get_current_active_user)
):
    """Batch feedback submission"""
    try:
        if not feedback_entries or len(feedback_entries) == 0:
            ErrorHandler.raise_bad_request("No feedback entries provided")
        
        if len(feedback_entries) > 50:
            ErrorHandler.raise_bad_request("Too many feedback entries (max 50)")
        
        # Validate all entries
        for i, entry in enumerate(feedback_entries):
            if "article_id" not in entry:
                ErrorHandler.raise_bad_request(f"Entry {i} missing article_id")
            
            if "feedback_score" in entry and not (0.0 <= entry["feedback_score"] <= 1.0):
                ErrorHandler.raise_bad_request(f"Entry {i} feedback_score must be between 0.0 and 1.0")
        
        # Process feedback entries in parallel
        tasks = []
        for entry in feedback_entries:
            task = submit_feedback_advanced(
                article_id=entry["article_id"],
                clicked=entry.get("clicked"),
                upvote=entry.get("upvote"),
                correct_after_days=entry.get("correct_after_days"),
                feedback_score=entry.get("feedback_score"),
                feedback_text=entry.get("feedback_text"),
                current_user=current_user
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_submissions = []
        failed_submissions = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_submissions.append({
                    "index": i,
                    "error": str(result)
                })
            else:
                successful_submissions.append(result)
        
        return {
            "total_entries": len(feedback_entries),
            "successful_submissions": len(successful_submissions),
            "failed_submissions": len(failed_submissions),
            "results": successful_submissions,
            "failures": failed_submissions
        }
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Batch feedback submission failed: {e}")
        ErrorHandler.raise_internal_server_error("Batch feedback submission failed", str(e))

@router.get(
    "/digests",
    summary="Enhanced Digest Retrieval",
    description="Get digests with advanced filtering and ML-powered insights",
    responses={
        200: {"description": "Digests retrieved successfully"},
        500: {"description": "Failed to retrieve digests", "model": ErrorResponse}
    }
)
async def get_digests_enhanced(
    limit: int = Query(default=10, description="Number of digests to return", ge=1, le=100),
    offset: int = Query(default=0, description="Number of digests to skip", ge=0),
    topic: Optional[str] = Query(default=None, description="Filter by topic"),
    include_analysis: bool = Query(default=True, description="Include ML analysis data"),
    current_user: User = Depends(get_current_active_user)
):
    """Enhanced digest retrieval with ML insights"""
    try:
        from .. import db
        
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                # Build query with optional topic filter
                query = """
                    SELECT id, title, content, topic, created_at, 
                           article_count, sentiment_summary, key_insights, analysis_metadata
                    FROM digests
                    WHERE 1=1
                """
                params = []
                
                if topic:
                    query += " AND topic = %s"
                    params.append(topic)
                
                query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
                params.extend([limit, offset])
                
                cur.execute(query, params)
                digest_rows = cur.fetchall()
                
                # Get total count
                count_query = "SELECT COUNT(*) FROM digests WHERE 1=1"
                count_params = []
                if topic:
                    count_query += " AND topic = %s"
                    count_params.append(topic)
                
                cur.execute(count_query, count_params)
                total_count = cur.fetchone()[0]
                
                digests = []
                for row in digest_rows:
                    digest_data = {
                        "id": str(row[0]),
                        "title": row[1],
                        "content": row[2],
                        "topic": row[3],
                        "created_at": row[4].isoformat(),
                        "article_count": row[5],
                        "sentiment_summary": json.loads(row[6]) if row[6] else {},
                        "key_insights": json.loads(row[7]) if row[7] else []
                    }
                    
                    # Include analysis metadata if requested
                    if include_analysis and row[8]:
                        digest_data["analysis_metadata"] = json.loads(row[8])
                    
                    digests.append(digest_data)
        
        return {
            "digests": digests,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "topic_filter": topic,
            "include_analysis": include_analysis
        }
        
    except Exception as e:
        logger.error(f"Failed to get digests: {e}")
        ErrorHandler.raise_internal_server_error("Failed to retrieve digests", str(e))

@router.get(
    "/recommendations",
    summary="Content Recommendations",
    description="Get personalized content recommendations based on user preferences",
    responses={
        200: {"description": "Recommendations retrieved successfully"},
        500: {"description": "Failed to generate recommendations", "model": ErrorResponse}
    }
)
async def get_content_recommendations(
    limit: int = Query(default=10, description="Number of recommendations", ge=1, le=50),
    current_user: User = Depends(get_current_active_user)
):
    """Get personalized content recommendations"""
    try:
        recommendations = await feedback_service.generate_content_recommendations(
            user_id=current_user.id,
            limit=limit
        )
        
        return {
            "recommendations": recommendations,
            "count": len(recommendations),
            "user_id": current_user.id,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}")
        ErrorHandler.raise_internal_server_error("Failed to generate recommendations", str(e))

@router.get(
    "/preferences",
    summary="User Preferences",
    description="Get user preferences based on feedback history",
    responses={
        200: {"description": "Preferences retrieved successfully"},
        500: {"description": "Failed to retrieve preferences", "model": ErrorResponse}
    }
)
async def get_user_preferences(
    current_user: User = Depends(get_current_active_user)
):
    """Get user preferences"""
    try:
        preferences = await feedback_service.generate_user_preferences(current_user.id)
        
        return {
            "preferences": preferences,
            "user_id": current_user.id,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get user preferences: {e}")
        ErrorHandler.raise_internal_server_error("Failed to retrieve preferences", str(e))

@router.post(
    "/digests/generate",
    summary="ML-Powered Digest Generation",
    description="Generate advanced digest using ML analysis, topic modeling, and sentiment trends",
    responses={
        200: {"description": "Digest generated successfully"},
        404: {"description": "No articles found", "model": ErrorResponse},
        500: {"description": "Digest generation failed", "model": ErrorResponse}
    }
)
async def generate_digest_ml(
    topic: Optional[str] = Query(default=None, description="Filter by specific topic"),
    days_back: int = Query(default=7, description="Number of days to analyze", ge=1, le=30),
    include_clustering: bool = Query(default=True, description="Include content clustering analysis"),
    include_sentiment_trends: bool = Query(default=True, description="Include sentiment trend analysis"),
    current_user: User = Depends(get_current_active_user)
):
    """Generate ML-powered digest with advanced analysis"""
    try:
        result = await feedback_service.generate_ml_digest(
            topic=topic,
            days_back=days_back,
            user_id=current_user.id
        )
        
        return {
            "message": "ML-powered digest generated successfully",
            "digest_id": result["digest_id"],
            "title": result["title"],
            "article_count": result["article_count"],
            "topic": result["topic"],
            "insights": result["insights"],
            "sentiment_analysis": result["sentiment_analysis"],
            "topics": result["topics"],
            "generated_at": result["generated_at"]
        }
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"ML digest generation failed: {e}")
        ErrorHandler.raise_internal_server_error("ML digest generation failed", str(e))

# Add missing methods to AdvancedFeedbackService
async def _invalidate_user_preferences_cache(self, user_id: int):
    """Invalidate user preferences cache"""
    try:
        cache_key = f"user_preferences:{user_id}"
        await cache.delete(cache_key)
    except Exception as e:
        logger.warning(f"Failed to invalidate user preferences cache: {e}")

# Add the method to the class
AdvancedFeedbackService._invalidate_user_preferences_cache = _invalidate_user_preferences_cache

# Legacy endpoints for backward compatibility
@router.post(
    "/submit/legacy",
    summary="Legacy Feedback Submission",
    description="Legacy endpoint for backward compatibility",
    deprecated=True
)
async def submit_feedback_legacy(
    article_id: int,
    clicked: Optional[bool] = None,
    upvote: Optional[bool] = None,
    correct_after_days: Optional[bool] = None,
    feedback_score: Optional[float] = None,
    feedback_text: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Legacy feedback submission endpoint"""
    return await submit_feedback_advanced(
        article_id, clicked, upvote, correct_after_days, 
        feedback_score, feedback_text, current_user
    )

@router.get(
    "/digests/legacy",
    summary="Legacy Digest Retrieval",
    description="Legacy endpoint for backward compatibility",
    deprecated=True
)
async def get_digests_legacy(
    limit: int = 10,
    offset: int = 0,
    topic: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Legacy digest retrieval endpoint"""
    return await get_digests_enhanced(limit, offset, topic, True, current_user)

@router.post(
    "/digests/generate/legacy",
    summary="Legacy Digest Generation",
    description="Legacy endpoint for backward compatibility",
    deprecated=True
)
async def generate_digest_legacy(
    topic: Optional[str] = None,
    days_back: int = 7,
    current_user: User = Depends(get_current_active_user)
):
    """Legacy digest generation endpoint"""
    return await generate_digest_ml(topic, days_back, True, True, current_user)
