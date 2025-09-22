-- Enhanced Monitoring and Quality Schema
-- Additional tables for enhanced features implemented in routers
\connect osint

-- Source quality scoring and health monitoring
CREATE TABLE IF NOT EXISTS source_quality_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id INTEGER REFERENCES sources(id) ON DELETE CASCADE,
    reliability_score FLOAT CHECK (reliability_score >= 0.0 AND reliability_score <= 1.0),
    performance_score FLOAT CHECK (performance_score >= 0.0 AND performance_score <= 1.0),
    content_quality_score FLOAT CHECK (content_quality_score >= 0.0 AND content_quality_score <= 1.0),
    uptime_score FLOAT CHECK (uptime_score >= 0.0 AND uptime_score <= 1.0),
    overall_score FLOAT CHECK (overall_score >= 0.0 AND overall_score <= 1.0),
    last_checked TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_id)
);

-- Source health check history
CREATE TABLE IF NOT EXISTS source_health_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id INTEGER REFERENCES sources(id) ON DELETE CASCADE,
    check_type VARCHAR(50) NOT NULL, -- 'availability', 'response_time', 'content_quality', 'reliability'
    check_result BOOLEAN NOT NULL,
    response_time_ms INTEGER,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    checked_at TIMESTAMPTZ DEFAULT NOW()
);

-- WebSocket message queuing and persistence
CREATE TABLE IF NOT EXISTS websocket_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    message_type VARCHAR(50) NOT NULL, -- 'notification', 'alert', 'update', 'status'
    priority INTEGER DEFAULT 0, -- 0=low, 1=medium, 2=high, 3=urgent
    message_data JSONB NOT NULL,
    expires_at TIMESTAMPTZ,
    is_delivered BOOLEAN DEFAULT FALSE,
    delivery_attempts INTEGER DEFAULT 0,
    last_delivery_attempt TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Performance metrics tracking
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name VARCHAR(100) NOT NULL, -- 'analysis', 'bandit', 'feedback', 'intelligence', 'sources'
    metric_name VARCHAR(100) NOT NULL, -- 'processing_time', 'cache_hit_rate', 'error_rate', 'throughput'
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(20), -- 'ms', 'percent', 'count', 'bytes'
    metadata JSONB DEFAULT '{}',
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

-- Circuit breaker state tracking
CREATE TABLE IF NOT EXISTS circuit_breaker_state (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name VARCHAR(100) NOT NULL UNIQUE,
    state VARCHAR(20) NOT NULL CHECK (state IN ('CLOSED', 'OPEN', 'HALF_OPEN')),
    failure_count INTEGER DEFAULT 0,
    failure_threshold INTEGER DEFAULT 5,
    success_count INTEGER DEFAULT 0,
    last_failure TIMESTAMPTZ,
    last_success TIMESTAMPTZ,
    next_attempt TIMESTAMPTZ,
    timeout_duration INTERVAL DEFAULT '60 seconds',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enhanced error monitoring and logging
CREATE TABLE IF NOT EXISTS error_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name VARCHAR(100) NOT NULL,
    error_type VARCHAR(100) NOT NULL, -- 'APIError', 'ValidationError', 'DatabaseError', 'TimeoutError'
    error_code VARCHAR(50),
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    request_id VARCHAR(100),
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    severity VARCHAR(20) DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    context_data JSONB DEFAULT '{}',
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Analysis performance monitoring
CREATE TABLE IF NOT EXISTS analysis_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_type VARCHAR(50) NOT NULL, -- 'sentiment', 'bias', 'stance', 'entity', 'topic'
    processing_time_ms INTEGER NOT NULL,
    cache_hit BOOLEAN DEFAULT FALSE,
    model_name VARCHAR(100),
    input_size INTEGER, -- character count
    output_size INTEGER, -- result count
    confidence_score FLOAT,
    error_occurred BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

-- Bandit algorithm performance tracking
CREATE TABLE IF NOT EXISTS bandit_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    algorithm_name VARCHAR(50) NOT NULL, -- 'UCB', 'Thompson', 'LinUCB', 'EpsilonGreedy'
    key_name VARCHAR(100) NOT NULL,
    selection_time_ms INTEGER,
    reward FLOAT,
    regret FLOAT,
    confidence FLOAT,
    context_size INTEGER,
    metadata JSONB DEFAULT '{}',
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

-- User preference tracking for recommendations
CREATE TABLE IF NOT EXISTS user_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    preference_type VARCHAR(50) NOT NULL, -- 'topic', 'source', 'sentiment', 'bias'
    preference_value VARCHAR(100) NOT NULL,
    preference_weight FLOAT DEFAULT 1.0,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, preference_type, preference_value)
);

-- ML digest generation tracking
CREATE TABLE IF NOT EXISTS digest_generation_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    digest_id UUID REFERENCES digests(id) ON DELETE CASCADE,
    generation_method VARCHAR(50) NOT NULL, -- 'lda', 'kmeans', 'tfidf', 'hybrid'
    processing_time_ms INTEGER,
    article_count INTEGER,
    topic_count INTEGER,
    sentiment_analysis_time_ms INTEGER,
    clustering_time_ms INTEGER,
    content_extraction_time_ms INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Intelligence threat detection logs
CREATE TABLE IF NOT EXISTS threat_detection_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    intelligence_data_id UUID REFERENCES intelligence_data(id) ON DELETE CASCADE,
    threat_type VARCHAR(50) NOT NULL, -- 'security', 'privacy', 'compliance', 'content'
    threat_level VARCHAR(20) NOT NULL CHECK (threat_level IN ('low', 'medium', 'high', 'critical')),
    detection_method VARCHAR(50) NOT NULL, -- 'pattern_matching', 'ml_analysis', 'rule_based'
    confidence_score FLOAT CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    threat_description TEXT,
    mitigation_suggestions TEXT[],
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    detected_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_source_quality_scores_source ON source_quality_scores (source_id);
CREATE INDEX IF NOT EXISTS idx_source_quality_scores_overall ON source_quality_scores (overall_score DESC);
CREATE INDEX IF NOT EXISTS idx_source_quality_scores_checked ON source_quality_scores (last_checked DESC);

CREATE INDEX IF NOT EXISTS idx_source_health_checks_source ON source_health_checks (source_id);
CREATE INDEX IF NOT EXISTS idx_source_health_checks_type ON source_health_checks (check_type);
CREATE INDEX IF NOT EXISTS idx_source_health_checks_result ON source_health_checks (check_result);
CREATE INDEX IF NOT EXISTS idx_source_health_checks_checked ON source_health_checks (checked_at DESC);

CREATE INDEX IF NOT EXISTS idx_websocket_messages_user ON websocket_messages (user_id);
CREATE INDEX IF NOT EXISTS idx_websocket_messages_type ON websocket_messages (message_type);
CREATE INDEX IF NOT EXISTS idx_websocket_messages_priority ON websocket_messages (priority DESC);
CREATE INDEX IF NOT EXISTS idx_websocket_messages_delivered ON websocket_messages (is_delivered);
CREATE INDEX IF NOT EXISTS idx_websocket_messages_expires ON websocket_messages (expires_at);
CREATE INDEX IF NOT EXISTS idx_websocket_messages_created ON websocket_messages (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_service ON performance_metrics (service_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_name ON performance_metrics (metric_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_recorded ON performance_metrics (recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_composite ON performance_metrics (service_name, metric_name, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_circuit_breaker_state_service ON circuit_breaker_state (service_name);
CREATE INDEX IF NOT EXISTS idx_circuit_breaker_state_state ON circuit_breaker_state (state);
CREATE INDEX IF NOT EXISTS idx_circuit_breaker_state_updated ON circuit_breaker_state (updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_error_logs_service ON error_logs (service_name);
CREATE INDEX IF NOT EXISTS idx_error_logs_type ON error_logs (error_type);
CREATE INDEX IF NOT EXISTS idx_error_logs_severity ON error_logs (severity);
CREATE INDEX IF NOT EXISTS idx_error_logs_resolved ON error_logs (resolved);
CREATE INDEX IF NOT EXISTS idx_error_logs_created ON error_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_error_logs_user ON error_logs (user_id);

CREATE INDEX IF NOT EXISTS idx_analysis_performance_type ON analysis_performance (analysis_type);
CREATE INDEX IF NOT EXISTS idx_analysis_performance_time ON analysis_performance (processing_time_ms);
CREATE INDEX IF NOT EXISTS idx_analysis_performance_cache ON analysis_performance (cache_hit);
CREATE INDEX IF NOT EXISTS idx_analysis_performance_recorded ON analysis_performance (recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_bandit_performance_algorithm ON bandit_performance (algorithm_name);
CREATE INDEX IF NOT EXISTS idx_bandit_performance_key ON bandit_performance (key_name);
CREATE INDEX IF NOT EXISTS idx_bandit_performance_recorded ON bandit_performance (recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_user_preferences_user ON user_preferences (user_id);
CREATE INDEX IF NOT EXISTS idx_user_preferences_type ON user_preferences (preference_type);
CREATE INDEX IF NOT EXISTS idx_user_preferences_updated ON user_preferences (last_updated DESC);

CREATE INDEX IF NOT EXISTS idx_digest_generation_logs_digest ON digest_generation_logs (digest_id);
CREATE INDEX IF NOT EXISTS idx_digest_generation_logs_method ON digest_generation_logs (generation_method);
CREATE INDEX IF NOT EXISTS idx_digest_generation_logs_created ON digest_generation_logs (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_threat_detection_logs_data ON threat_detection_logs (intelligence_data_id);
CREATE INDEX IF NOT EXISTS idx_threat_detection_logs_type ON threat_detection_logs (threat_type);
CREATE INDEX IF NOT EXISTS idx_threat_detection_logs_level ON threat_detection_logs (threat_level);
CREATE INDEX IF NOT EXISTS idx_threat_detection_logs_resolved ON threat_detection_logs (is_resolved);
CREATE INDEX IF NOT EXISTS idx_threat_detection_logs_detected ON threat_detection_logs (detected_at DESC);

-- Create materialized views for monitoring dashboards
CREATE MATERIALIZED VIEW IF NOT EXISTS monitoring_dashboard_stats AS
SELECT 
    DATE(recorded_at) as date,
    service_name,
    COUNT(*) as total_metrics,
    AVG(metric_value) as avg_value,
    MIN(metric_value) as min_value,
    MAX(metric_value) as max_value,
    STDDEV(metric_value) as stddev_value
FROM performance_metrics
WHERE recorded_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE(recorded_at), service_name
ORDER BY date DESC, service_name;

CREATE MATERIALIZED VIEW IF NOT EXISTS error_summary_stats AS
SELECT 
    DATE(created_at) as date,
    service_name,
    error_type,
    severity,
    COUNT(*) as error_count,
    COUNT(CASE WHEN resolved = TRUE THEN 1 END) as resolved_count
FROM error_logs
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at), service_name, error_type, severity
ORDER BY date DESC, error_count DESC;

CREATE MATERIALIZED VIEW IF NOT EXISTS source_health_summary AS
SELECT 
    s.id as source_id,
    s.name as source_name,
    sqs.overall_score,
    sqs.last_checked,
    COUNT(shc.id) as total_checks,
    COUNT(CASE WHEN shc.check_result = TRUE THEN 1 END) as successful_checks,
    AVG(shc.response_time_ms) as avg_response_time
FROM sources s
LEFT JOIN source_quality_scores sqs ON s.id = sqs.source_id
LEFT JOIN source_health_checks shc ON s.id = shc.source_id
WHERE shc.checked_at >= NOW() - INTERVAL '7 days' OR shc.checked_at IS NULL
GROUP BY s.id, s.name, sqs.overall_score, sqs.last_checked
ORDER BY sqs.overall_score DESC NULLS LAST;

-- Create indexes on materialized views
CREATE INDEX IF NOT EXISTS idx_monitoring_dashboard_stats_date ON monitoring_dashboard_stats (date);
CREATE INDEX IF NOT EXISTS idx_monitoring_dashboard_stats_service ON monitoring_dashboard_stats (service_name);
CREATE INDEX IF NOT EXISTS idx_error_summary_stats_date ON error_summary_stats (date);
CREATE INDEX IF NOT EXISTS idx_error_summary_stats_service ON error_summary_stats (service_name);
CREATE INDEX IF NOT EXISTS idx_source_health_summary_score ON source_health_summary (overall_score DESC);

-- Create functions for monitoring
CREATE OR REPLACE FUNCTION refresh_monitoring_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY monitoring_dashboard_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY error_summary_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY source_health_summary;
END;
$$ LANGUAGE plpgsql;

-- Create function to clean up expired data
CREATE OR REPLACE FUNCTION cleanup_expired_monitoring_data()
RETURNS void AS $$
BEGIN
    -- Clean up expired WebSocket messages
    DELETE FROM websocket_messages 
    WHERE expires_at < NOW() AND is_delivered = TRUE;
    
    -- Clean up old performance metrics (keep 90 days)
    DELETE FROM performance_metrics 
    WHERE recorded_at < NOW() - INTERVAL '90 days';
    
    -- Clean up old error logs (keep 30 days)
    DELETE FROM error_logs 
    WHERE created_at < NOW() - INTERVAL '30 days' AND resolved = TRUE;
    
    -- Clean up old analysis performance logs (keep 60 days)
    DELETE FROM analysis_performance 
    WHERE recorded_at < NOW() - INTERVAL '60 days';
    
    -- Clean up old bandit performance logs (keep 60 days)
    DELETE FROM bandit_performance 
    WHERE recorded_at < NOW() - INTERVAL '60 days';
END;
$$ LANGUAGE plpgsql;

-- Create function to get system health status
CREATE OR REPLACE FUNCTION get_system_health_status()
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'overall_health', CASE 
            WHEN error_count = 0 THEN 'healthy'
            WHEN error_count < 10 THEN 'warning'
            ELSE 'critical'
        END,
        'error_count_24h', error_count,
        'active_circuit_breakers', circuit_breaker_count,
        'avg_response_time_ms', avg_response_time,
        'source_health_score', source_health_avg,
        'last_updated', NOW()
    ) INTO result
    FROM (
        SELECT 
            COUNT(*) as error_count,
            (SELECT COUNT(*) FROM circuit_breaker_state WHERE state != 'CLOSED') as circuit_breaker_count,
            (SELECT AVG(response_time_ms) FROM source_health_checks WHERE checked_at >= NOW() - INTERVAL '1 hour') as avg_response_time,
            (SELECT AVG(overall_score) FROM source_quality_scores WHERE last_checked >= NOW() - INTERVAL '1 hour') as source_health_avg
        FROM error_logs 
        WHERE created_at >= NOW() - INTERVAL '24 hours' AND severity IN ('high', 'critical')
    ) stats;
    
    RETURN COALESCE(result, '{"overall_health": "unknown", "error_count_24h": 0, "active_circuit_breakers": 0, "avg_response_time_ms": 0, "source_health_score": 0, "last_updated": "' || NOW() || '"}'::json);
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON source_quality_scores TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON source_health_checks TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON websocket_messages TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON performance_metrics TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON circuit_breaker_state TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON error_logs TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON analysis_performance TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON bandit_performance TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON user_preferences TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON digest_generation_logs TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON threat_detection_logs TO osint_user;

GRANT SELECT ON monitoring_dashboard_stats TO osint_user;
GRANT SELECT ON error_summary_stats TO osint_user;
GRANT SELECT ON source_health_summary TO osint_user;

GRANT EXECUTE ON FUNCTION refresh_monitoring_views() TO osint_user;
GRANT EXECUTE ON FUNCTION cleanup_expired_monitoring_data() TO osint_user;
GRANT EXECUTE ON FUNCTION get_system_health_status() TO osint_user;
