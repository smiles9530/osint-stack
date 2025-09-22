-- Machine Learning and Analytics Schema
\connect osint

-- Article features table
CREATE TABLE IF NOT EXISTS article_features (
    article_id BIGINT PRIMARY KEY,
    word_count INTEGER,
    char_count INTEGER,
    sentence_count INTEGER,
    avg_word_length NUMERIC,
    unique_words INTEGER,
    exclamation_count INTEGER,
    question_count INTEGER,
    uppercase_ratio NUMERIC,
    digit_ratio NUMERIC,
    url_count INTEGER,
    email_count INTEGER,
    phone_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Sentiment analysis table
CREATE TABLE IF NOT EXISTS article_sentiment (
    article_id BIGINT PRIMARY KEY,
    vader_compound NUMERIC,
    vader_pos NUMERIC,
    vader_neu NUMERIC,
    vader_neg NUMERIC,
    custom_pos NUMERIC,
    custom_neg NUMERIC,
    custom_neu NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Entity extraction table
CREATE TABLE IF NOT EXISTS article_entities (
    id SERIAL PRIMARY KEY,
    article_id BIGINT,
    entity_type VARCHAR(50) NOT NULL,
    entity_name TEXT NOT NULL,
    confidence NUMERIC DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(article_id, entity_type, entity_name)
);

-- ML topics table
CREATE TABLE IF NOT EXISTS article_topics_ml (
    article_id BIGINT,
    topic_name VARCHAR(100) NOT NULL,
    topic_score NUMERIC,
    keywords JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (article_id, topic_name)
);

-- Anomaly detection table
CREATE TABLE IF NOT EXISTS article_anomalies (
    id SERIAL PRIMARY KEY,
    article_id BIGINT,
    anomaly_type VARCHAR(50) NOT NULL,
    anomaly_score NUMERIC,
    severity VARCHAR(20) DEFAULT 'medium',
    description TEXT,
    features JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    is_resolved BOOLEAN DEFAULT FALSE
);

-- Real-time analytics cache table
CREATE TABLE IF NOT EXISTS analytics_cache (
    cache_key VARCHAR(255) PRIMARY KEY,
    cache_data JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Dashboard configurations per user
CREATE TABLE IF NOT EXISTS dashboard_configs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    dashboard_name VARCHAR(100) NOT NULL,
    config_data JSONB NOT NULL,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, dashboard_name)
);

-- Real-time alerts table
CREATE TABLE IF NOT EXISTS real_time_alerts (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    data JSONB,
    target_users INTEGER[],
    target_roles VARCHAR(50)[],
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    acknowledged_by INTEGER REFERENCES users(id),
    acknowledged_at TIMESTAMPTZ
);

-- WebSocket connections tracking
CREATE TABLE IF NOT EXISTS websocket_connections (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    connection_id VARCHAR(255) UNIQUE NOT NULL,
    connected_at TIMESTAMPTZ DEFAULT NOW(),
    last_ping TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_article_features_word_count ON article_features (word_count);
CREATE INDEX IF NOT EXISTS idx_article_sentiment_vader_compound ON article_sentiment (vader_compound);
CREATE INDEX IF NOT EXISTS idx_article_entities_type ON article_entities (entity_type);
CREATE INDEX IF NOT EXISTS idx_article_entities_name ON article_entities (entity_name);
CREATE INDEX IF NOT EXISTS idx_article_topics_ml_score ON article_topics_ml (topic_score DESC);
CREATE INDEX IF NOT EXISTS idx_article_anomalies_type ON article_anomalies (anomaly_type);
CREATE INDEX IF NOT EXISTS idx_article_anomalies_severity ON article_anomalies (severity);
CREATE INDEX IF NOT EXISTS idx_article_anomalies_created ON article_anomalies (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_cache_expires ON analytics_cache (expires_at);
CREATE INDEX IF NOT EXISTS idx_dashboard_configs_user ON dashboard_configs (user_id);
CREATE INDEX IF NOT EXISTS idx_real_time_alerts_active ON real_time_alerts (is_active, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_real_time_alerts_type ON real_time_alerts (alert_type);
CREATE INDEX IF NOT EXISTS idx_websocket_connections_user ON websocket_connections (user_id);
CREATE INDEX IF NOT EXISTS idx_websocket_connections_active ON websocket_connections (is_active);

-- Create materialized view for dashboard statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS dashboard_stats AS
SELECT 
    DATE(a.fetched_at) as date,
    COUNT(*) as total_articles,
    AVG(s.vader_compound) as avg_sentiment,
    COUNT(DISTINCT a.source_id) as active_sources,
    COUNT(DISTINCT e.entity_name) as unique_entities,
    COUNT(DISTINCT t.topic_name) as unique_topics
FROM articles a
LEFT JOIN article_sentiment s ON a.id = s.article_id
LEFT JOIN article_entities e ON a.id = e.article_id
LEFT JOIN article_topics_ml t ON a.id = t.article_id
WHERE a.fetched_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE(a.fetched_at)
ORDER BY date DESC;

-- Create index on materialized view
CREATE INDEX IF NOT EXISTS idx_dashboard_stats_date ON dashboard_stats (date);

-- Create function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_dashboard_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY dashboard_stats;
END;
$$ LANGUAGE plpgsql;

-- Create function to clean up expired cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS void AS $$
BEGIN
    DELETE FROM analytics_cache WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- Create function to get real-time statistics
CREATE OR REPLACE FUNCTION get_realtime_stats()
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'total_articles', (SELECT COUNT(*) FROM articles),
        'articles_today', (SELECT COUNT(*) FROM articles WHERE DATE(fetched_at) = CURRENT_DATE),
        'active_sources', (SELECT COUNT(DISTINCT source_id) FROM articles WHERE source_id IS NOT NULL),
        'avg_sentiment', (SELECT AVG(vader_compound) FROM article_sentiment WHERE created_at >= NOW() - INTERVAL '24 hours'),
        'active_connections', (SELECT COUNT(*) FROM websocket_connections WHERE is_active = TRUE),
        'pending_alerts', (SELECT COUNT(*) FROM real_time_alerts WHERE is_active = TRUE AND (expires_at IS NULL OR expires_at > NOW()))
    ) INTO result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Intelligence processing table for tracking collection workflows
CREATE TABLE IF NOT EXISTS intelligence_processing (
    id SERIAL PRIMARY KEY,
    domain VARCHAR(50) NOT NULL,
    source_count INTEGER DEFAULT 0,
    processing_type VARCHAR(100),
    processing_timestamp TIMESTAMPTZ,
    user_id INTEGER,
    raw_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Intelligence synthesis table for storing AI analysis results
CREATE TABLE IF NOT EXISTS intelligence_synthesis (
    id SERIAL PRIMARY KEY,
    collection_cycle VARCHAR(100),
    synthesis_data JSONB,
    processing_count INTEGER DEFAULT 0,
    synthesis_timestamp TIMESTAMPTZ,
    user_id INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Indexes for intelligence tables
CREATE INDEX IF NOT EXISTS idx_intelligence_processing_domain ON intelligence_processing(domain);
CREATE INDEX IF NOT EXISTS idx_intelligence_processing_timestamp ON intelligence_processing(processing_timestamp);
CREATE INDEX IF NOT EXISTS idx_intelligence_synthesis_cycle ON intelligence_synthesis(collection_cycle);
CREATE INDEX IF NOT EXISTS idx_intelligence_synthesis_timestamp ON intelligence_synthesis(synthesis_timestamp);