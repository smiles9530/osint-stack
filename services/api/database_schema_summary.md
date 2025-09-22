# Database Schema Summary

## Overview
The OSINT Stack database schema is comprehensive and well-structured, supporting advanced ML-powered analysis, real-time processing, and comprehensive monitoring. The schema is organized across 11 initialization files covering different aspects of the system.

## Schema Files

### Core Infrastructure
- **`001-extensions.sql`** - PostgreSQL extensions (uuid-ossp, pgvector, timescaledb)
- **`010-schema.sql`** - Core tables (users, articles, sources, embeddings, topics)
- **`020-init-users.sql`** - User initialization and seeding
- **`030-partition-management.sql`** - Article table partitioning for performance

### Machine Learning & Analytics
- **`040-ml-schema.sql`** - ML features, sentiment analysis, entity extraction, anomaly detection
- **`050-bandit-seed.sql`** - Bandit algorithm initialization
- **`060-ml-analysis.sql`** - ML analysis tables and functions
- **`070-entity-schema.sql`** - Entity extraction and knowledge graph
- **`080-stance-sentiment-schema.sql`** - Advanced stance, sentiment, and bias analysis

### Enhanced Features
- **`090-optimized-article-schema.sql`** - Optimized article schema with analysis results
- **`100-enhanced-features-schema.sql`** - Enhanced features (bandit, feedback, intelligence)
- **`110-enhanced-monitoring-schema.sql`** - Monitoring, quality scoring, and observability

## Table Categories

### 🔐 **Authentication & Users**
- `users` - User accounts and profiles
- `user_sessions` - JWT token management
- `user_preferences` - User preference tracking for recommendations

### 📰 **Content Management**
- `articles` - Main article storage (partitioned by date)
- `sources` - News source information
- `topics` - Topic classification
- `article_topics` - Article-topic relationships
- `digests` - ML-generated content digests

### 🤖 **Machine Learning**
- `article_features` - Text analysis features
- `article_sentiment` - Sentiment analysis results
- `article_entities` - Named entity recognition
- `article_topics_ml` - ML topic modeling
- `article_anomalies` - Anomaly detection results
- `embeddings` - Vector embeddings (pgvector)

### 📊 **Analysis & Intelligence**
- `stance_sentiment_analysis` - Chunk-level analysis
- `aggregated_analysis` - Article-level aggregated results
- `source_analysis_daily` - Daily source analysis rollups
- `topic_analysis_daily` - Daily topic analysis rollups
- `intelligence_data` - Intelligence processing data
- `intelligence_insights` - Intelligence analysis insights

### 🎯 **Bandit Algorithms**
- `bandit_selections` - Algorithm selections
- `bandit_rewards` - Reward feedback
- `bandit_state` - Algorithm state tracking
- `bandit_performance` - Performance metrics

### 📈 **Monitoring & Quality**
- `source_quality_scores` - Source health scoring
- `source_health_checks` - Health check history
- `performance_metrics` - System performance tracking
- `circuit_breaker_state` - Circuit breaker states
- `error_logs` - Error tracking and logging
- `analysis_performance` - Analysis performance metrics

### 🔔 **Real-time Features**
- `websocket_messages` - Message queuing and persistence
- `websocket_connections` - Active WebSocket connections
- `real_time_alerts` - Real-time alert system
- `analysis_alerts` - Analysis-based alerts
- `dashboard_updates` - Dashboard update tracking

### 🗄️ **Caching & Optimization**
- `analytics_cache` - Analytics result caching
- `news_seen` - URL deduplication
- `processing_errors` - Error handling and retry

## Key Features

### ✅ **Performance Optimizations**
- **Table Partitioning**: Articles partitioned by date for better query performance
- **Comprehensive Indexing**: 50+ indexes for optimal query performance
- **Materialized Views**: Pre-computed analytics for dashboard performance
- **JSONB Storage**: Flexible JSON storage for complex data structures

### ✅ **Advanced Analytics**
- **Time Series Data**: TimescaleDB integration for time-series analytics
- **Vector Search**: pgvector integration for semantic search
- **Knowledge Graph**: Entity relationships and connections
- **ML Pipeline**: Complete ML analysis pipeline with feature extraction

### ✅ **Real-time Capabilities**
- **WebSocket Support**: Real-time communication with message queuing
- **Alert System**: Configurable alerts with severity levels
- **Dashboard Updates**: Real-time dashboard metric updates
- **Circuit Breakers**: Fault tolerance and reliability patterns

### ✅ **Monitoring & Observability**
- **Performance Metrics**: Comprehensive performance tracking
- **Error Logging**: Structured error logging with context
- **Health Checks**: Source health monitoring and scoring
- **Quality Metrics**: Source quality and reliability scoring

## Database Statistics

### Table Count
- **Core Tables**: 15
- **ML/Analytics Tables**: 12
- **Monitoring Tables**: 8
- **Real-time Tables**: 6
- **Total Tables**: ~41

### Index Count
- **Primary Keys**: 41
- **Foreign Keys**: 25
- **Performance Indexes**: 50+
- **Composite Indexes**: 15+
- **GIN Indexes**: 8 (for JSONB)

### Materialized Views
- `article_stats` - Article statistics
- `dashboard_stats` - Dashboard metrics
- `analysis_dashboard_stats` - Analysis statistics
- `bandit_statistics` - Bandit algorithm stats
- `user_engagement_metrics` - User engagement
- `monitoring_dashboard_stats` - Monitoring metrics
- `error_summary_stats` - Error summaries
- `source_health_summary` - Source health overview

## Data Retention & Cleanup

### Automatic Cleanup Functions
- `cleanup_expired_cache()` - Analytics cache cleanup
- `cleanup_expired_monitoring_data()` - Monitoring data cleanup
- `refresh_article_stats()` - Materialized view refresh
- `refresh_monitoring_views()` - Monitoring view refresh

### Retention Policies
- **Performance Metrics**: 90 days
- **Error Logs**: 30 days (resolved), indefinite (unresolved)
- **Analysis Performance**: 60 days
- **WebSocket Messages**: Until delivered + expiration
- **Health Checks**: 7 days for active monitoring

## Security & Permissions

### User Roles
- `osint_user` - Application user with full CRUD permissions
- `osint_admin` - Administrative user (if needed)

### Data Protection
- **Cascade Deletes**: Proper cleanup on user deletion
- **Foreign Key Constraints**: Data integrity enforcement
- **Check Constraints**: Data validation at database level
- **Unique Constraints**: Prevent duplicate data

## Scalability Considerations

### Horizontal Scaling
- **Partitioning**: Articles table partitioned by date
- **Read Replicas**: Support for read-only replicas
- **Connection Pooling**: Database connection management

### Vertical Scaling
- **Index Optimization**: Comprehensive indexing strategy
- **Query Optimization**: Materialized views for complex queries
- **Memory Management**: Efficient data types and storage

## Future Enhancements

### Planned Additions
- **Data Archiving**: Long-term data archiving strategy
- **Backup Strategy**: Automated backup and recovery
- **Monitoring Alerts**: Database-level monitoring alerts
- **Performance Tuning**: Query performance optimization

### Schema Evolution
- **Migration Support**: Version-controlled schema changes
- **Backward Compatibility**: Safe schema updates
- **Feature Flags**: Database-level feature toggles

## Conclusion

The OSINT Stack database schema is a robust, scalable, and feature-rich foundation that supports:

- **Advanced ML Analysis**: Complete pipeline from data ingestion to analysis
- **Real-time Processing**: WebSocket communication and real-time updates
- **Comprehensive Monitoring**: Performance, health, and error tracking
- **High Performance**: Optimized for large-scale data processing
- **Reliability**: Fault tolerance and error handling
- **Observability**: Complete system monitoring and analytics

The schema is well-designed for the enhanced features implemented in the routers and provides a solid foundation for future development and scaling.
