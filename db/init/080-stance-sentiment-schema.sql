-- Stance, Sentiment, and Bias Analysis Schema
\connect osint

-- Chunk-level analysis results
CREATE TABLE IF NOT EXISTS stance_sentiment_analysis (
    id SERIAL PRIMARY KEY,
    article_id BIGINT NOT NULL,
    chunk_id VARCHAR(100) NOT NULL,
    chunk_text TEXT NOT NULL,
    sentiment_scores JSONB NOT NULL,  -- {positive: 0.3, negative: 0.2, neutral: 0.5}
    stance_scores JSONB NOT NULL,     -- {supports: 0.4, refutes: 0.3, neutral: 0.3}
    toxicity_scores JSONB NOT NULL,   -- {toxic: 0.1, non-toxic: 0.9}
    bias_scores JSONB NOT NULL,       -- {left: 0.2, center: 0.6, right: 0.2}
    confidence FLOAT NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(article_id, chunk_id)
);

-- Aggregated analysis results by article/source/topic
CREATE TABLE IF NOT EXISTS aggregated_analysis (
    id SERIAL PRIMARY KEY,
    article_id BIGINT UNIQUE NOT NULL,
    source_id VARCHAR(100),
    topic VARCHAR(200),
    total_chunks INTEGER NOT NULL DEFAULT 0,
    sentiment_distribution JSONB NOT NULL,  -- Average sentiment across chunks
    stance_distribution JSONB NOT NULL,     -- Average stance across chunks
    toxicity_levels JSONB NOT NULL,         -- Average toxicity across chunks
    bias_scores JSONB NOT NULL,             -- Average bias across chunks
    confidence_avg FLOAT NOT NULL DEFAULT 0.0,
    risk_flags JSONB DEFAULT '[]',          -- Array of risk flags
    trend_direction VARCHAR(50) DEFAULT 'neutral',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Source-level aggregated analysis (daily rollups)
CREATE TABLE IF NOT EXISTS source_analysis_daily (
    id SERIAL PRIMARY KEY,
    source_id VARCHAR(100) NOT NULL,
    analysis_date DATE NOT NULL,
    total_articles INTEGER DEFAULT 0,
    avg_sentiment JSONB,              -- Daily average sentiment
    avg_stance JSONB,                 -- Daily average stance
    avg_toxicity JSONB,               -- Daily average toxicity
    avg_bias JSONB,                   -- Daily average bias
    confidence_avg FLOAT DEFAULT 0.0,
    risk_flags_count JSONB DEFAULT '{}',  -- Count of each risk flag type
    trend_direction VARCHAR(50) DEFAULT 'stable',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_id, analysis_date)
);

-- Topic-level aggregated analysis (daily rollups)
CREATE TABLE IF NOT EXISTS topic_analysis_daily (
    id SERIAL PRIMARY KEY,
    topic VARCHAR(200) NOT NULL,
    analysis_date DATE NOT NULL,
    total_articles INTEGER DEFAULT 0,
    avg_sentiment JSONB,              -- Daily average sentiment
    avg_stance JSONB,                 -- Daily average stance
    avg_toxicity JSONB,               -- Daily average toxicity
    avg_bias JSONB,                   -- Daily average bias
    confidence_avg FLOAT DEFAULT 0.0,
    risk_flags_count JSONB DEFAULT '{}',  -- Count of each risk flag type
    trend_direction VARCHAR(50) DEFAULT 'stable',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(topic, analysis_date)
);

-- Analysis alerts and notifications
CREATE TABLE IF NOT EXISTS analysis_alerts (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(50) NOT NULL,  -- sentiment_shift, stance_change, toxicity_spike, extreme_bias
    severity VARCHAR(20) NOT NULL,    -- low, medium, high, critical
    source_id VARCHAR(100),
    topic VARCHAR(200),
    article_id BIGINT,
    message TEXT NOT NULL,
    alert_data JSONB,                 -- Additional alert context
    is_acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by INTEGER REFERENCES users(id),
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Analysis thresholds configuration
CREATE TABLE IF NOT EXISTS analysis_thresholds (
    id SERIAL PRIMARY KEY,
    threshold_name VARCHAR(100) UNIQUE NOT NULL,
    threshold_type VARCHAR(50) NOT NULL,  -- sentiment, stance, toxicity, bias
    threshold_value FLOAT NOT NULL,
    severity_level VARCHAR(20) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_stance_sentiment_article ON stance_sentiment_analysis (article_id);
CREATE INDEX IF NOT EXISTS idx_stance_sentiment_chunk ON stance_sentiment_analysis (chunk_id);
CREATE INDEX IF NOT EXISTS idx_stance_sentiment_created ON stance_sentiment_analysis (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_aggregated_analysis_article ON aggregated_analysis (article_id);
CREATE INDEX IF NOT EXISTS idx_aggregated_analysis_source ON aggregated_analysis (source_id);
CREATE INDEX IF NOT EXISTS idx_aggregated_analysis_topic ON aggregated_analysis (topic);
CREATE INDEX IF NOT EXISTS idx_aggregated_analysis_created ON aggregated_analysis (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_source_analysis_daily_source ON source_analysis_daily (source_id);
CREATE INDEX IF NOT EXISTS idx_source_analysis_daily_date ON source_analysis_daily (analysis_date DESC);
CREATE INDEX IF NOT EXISTS idx_source_analysis_daily_composite ON source_analysis_daily (source_id, analysis_date);

CREATE INDEX IF NOT EXISTS idx_topic_analysis_daily_topic ON topic_analysis_daily (topic);
CREATE INDEX IF NOT EXISTS idx_topic_analysis_daily_date ON topic_analysis_daily (analysis_date DESC);
CREATE INDEX IF NOT EXISTS idx_topic_analysis_daily_composite ON topic_analysis_daily (topic, analysis_date);

CREATE INDEX IF NOT EXISTS idx_analysis_alerts_type ON analysis_alerts (alert_type);
CREATE INDEX IF NOT EXISTS idx_analysis_alerts_severity ON analysis_alerts (severity);
CREATE INDEX IF NOT EXISTS idx_analysis_alerts_source ON analysis_alerts (source_id);
CREATE INDEX IF NOT EXISTS idx_analysis_alerts_topic ON analysis_alerts (topic);
CREATE INDEX IF NOT EXISTS idx_analysis_alerts_created ON analysis_alerts (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_alerts_acknowledged ON analysis_alerts (is_acknowledged);

CREATE INDEX IF NOT EXISTS idx_analysis_thresholds_type ON analysis_thresholds (threshold_type);
CREATE INDEX IF NOT EXISTS idx_analysis_thresholds_active ON analysis_thresholds (is_active);

-- Create materialized view for dashboard statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS analysis_dashboard_stats AS
SELECT 
    DATE(created_at) as analysis_date,
    COUNT(DISTINCT article_id) as total_articles_analyzed,
    COUNT(DISTINCT source_id) as sources_analyzed,
    COUNT(DISTINCT topic) as topics_analyzed,
    AVG(confidence_avg) as avg_confidence,
    COUNT(CASE WHEN jsonb_array_length(risk_flags) > 0 THEN 1 END) as articles_with_risks,
    COUNT(CASE WHEN trend_direction = 'positive' THEN 1 END) as positive_trends,
    COUNT(CASE WHEN trend_direction = 'negative' THEN 1 END) as negative_trends,
    COUNT(CASE WHEN trend_direction = 'supportive' THEN 1 END) as supportive_trends,
    COUNT(CASE WHEN trend_direction = 'refutational' THEN 1 END) as refutational_trends
FROM aggregated_analysis
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY analysis_date DESC;

-- Create index on materialized view
CREATE INDEX IF NOT EXISTS idx_analysis_dashboard_stats_date ON analysis_dashboard_stats (analysis_date);

-- Create function to refresh analysis dashboard stats
CREATE OR REPLACE FUNCTION refresh_analysis_dashboard_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY analysis_dashboard_stats;
END;
$$ LANGUAGE plpgsql;

-- Create function to get analysis summary for a source
CREATE OR REPLACE FUNCTION get_source_analysis_summary(p_source_id VARCHAR(100), p_days INTEGER DEFAULT 7)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'source_id', p_source_id,
        'analysis_period_days', p_days,
        'total_articles', COUNT(*),
        'avg_confidence', AVG(confidence_avg),
        'sentiment_distribution', json_object_agg('sentiment', sentiment_distribution),
        'stance_distribution', json_object_agg('stance', stance_distribution),
        'toxicity_levels', json_object_agg('toxicity', toxicity_levels),
        'bias_scores', json_object_agg('bias', bias_scores),
        'risk_flags_summary', json_object_agg('risk_flags', risk_flags),
        'trend_directions', json_object_agg('trend', trend_direction)
    ) INTO result
    FROM aggregated_analysis
    WHERE source_id = p_source_id 
    AND created_at >= NOW() - INTERVAL '%s days';
    
    RETURN COALESCE(result, '{}'::json);
END;
$$ LANGUAGE plpgsql;

-- Create function to get analysis summary for a topic
CREATE OR REPLACE FUNCTION get_topic_analysis_summary(p_topic VARCHAR(200), p_days INTEGER DEFAULT 7)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'topic', p_topic,
        'analysis_period_days', p_days,
        'total_articles', COUNT(*),
        'avg_confidence', AVG(confidence_avg),
        'sentiment_distribution', json_object_agg('sentiment', sentiment_distribution),
        'stance_distribution', json_object_agg('stance', stance_distribution),
        'toxicity_levels', json_object_agg('toxicity', toxicity_levels),
        'bias_scores', json_object_agg('bias', bias_scores),
        'risk_flags_summary', json_object_agg('risk_flags', risk_flags),
        'trend_directions', json_object_agg('trend', trend_direction)
    ) INTO result
    FROM aggregated_analysis
    WHERE topic = p_topic 
    AND created_at >= NOW() - INTERVAL '%s days';
    
    RETURN COALESCE(result, '{}'::json);
END;
$$ LANGUAGE plpgsql;

-- Create function to detect analysis anomalies
CREATE OR REPLACE FUNCTION detect_analysis_anomalies(p_hours INTEGER DEFAULT 24)
RETURNS TABLE(
    anomaly_type VARCHAR(50),
    source_id VARCHAR(100),
    topic VARCHAR(200),
    severity VARCHAR(20),
    description TEXT,
    anomaly_data JSONB
) AS $$
BEGIN
    -- Detect sudden sentiment shifts
    RETURN QUERY
    SELECT 
        'sentiment_shift'::VARCHAR(50) as anomaly_type,
        aa.source_id,
        aa.topic,
        CASE 
            WHEN (aa.sentiment_distribution->>'negative')::FLOAT > 0.7 THEN 'high'::VARCHAR(20)
            WHEN (aa.sentiment_distribution->>'negative')::FLOAT > 0.5 THEN 'medium'::VARCHAR(20)
            ELSE 'low'::VARCHAR(20)
        END as severity,
        'Sudden increase in negative sentiment detected'::TEXT as description,
        aa.sentiment_distribution as anomaly_data
    FROM aggregated_analysis aa
    WHERE aa.created_at >= NOW() - INTERVAL '%s hours'
    AND (aa.sentiment_distribution->>'negative')::FLOAT > 0.5;
    
    -- Detect stance changes
    RETURN QUERY
    SELECT 
        'stance_change'::VARCHAR(50) as anomaly_type,
        aa.source_id,
        aa.topic,
        CASE 
            WHEN (aa.stance_distribution->>'refutes')::FLOAT > 0.7 THEN 'high'::VARCHAR(20)
            WHEN (aa.stance_distribution->>'refutes')::FLOAT > 0.5 THEN 'medium'::VARCHAR(20)
            ELSE 'low'::VARCHAR(20)
        END as severity,
        'High refutation stance detected'::TEXT as description,
        aa.stance_distribution as anomaly_data
    FROM aggregated_analysis aa
    WHERE aa.created_at >= NOW() - INTERVAL '%s hours'
    AND (aa.stance_distribution->>'refutes')::FLOAT > 0.5;
    
    -- Detect toxicity spikes
    RETURN QUERY
    SELECT 
        'toxicity_spike'::VARCHAR(50) as anomaly_type,
        aa.source_id,
        aa.topic,
        CASE 
            WHEN (aa.toxicity_levels->>'toxic')::FLOAT > 0.3 THEN 'high'::VARCHAR(20)
            WHEN (aa.toxicity_levels->>'toxic')::FLOAT > 0.1 THEN 'medium'::VARCHAR(20)
            ELSE 'low'::VARCHAR(20)
        END as severity,
        'Toxicity spike detected'::TEXT as description,
        aa.toxicity_levels as anomaly_data
    FROM aggregated_analysis aa
    WHERE aa.created_at >= NOW() - INTERVAL '%s hours'
    AND (aa.toxicity_levels->>'toxic')::FLOAT > 0.1;
    
    -- Detect extreme bias
    RETURN QUERY
    SELECT 
        'extreme_bias'::VARCHAR(50) as anomaly_type,
        aa.source_id,
        aa.topic,
        CASE 
            WHEN GREATEST(
                (aa.bias_scores->>'left')::FLOAT,
                (aa.bias_scores->>'right')::FLOAT
            ) > 0.8 THEN 'high'::VARCHAR(20)
            WHEN GREATEST(
                (aa.bias_scores->>'left')::FLOAT,
                (aa.bias_scores->>'right')::FLOAT
            ) > 0.6 THEN 'medium'::VARCHAR(20)
            ELSE 'low'::VARCHAR(20)
        END as severity,
        'Extreme bias detected'::TEXT as description,
        aa.bias_scores as anomaly_data
    FROM aggregated_analysis aa
    WHERE aa.created_at >= NOW() - INTERVAL '%s hours'
    AND GREATEST(
        (aa.bias_scores->>'left')::FLOAT,
        (aa.bias_scores->>'right')::FLOAT
    ) > 0.6;
END;
$$ LANGUAGE plpgsql;

-- Insert default analysis thresholds
INSERT INTO analysis_thresholds (threshold_name, threshold_type, threshold_value, severity_level) VALUES
('sentiment_shift_medium', 'sentiment', 0.3, 'medium'),
('sentiment_shift_high', 'sentiment', 0.5, 'high'),
('stance_change_medium', 'stance', 0.4, 'medium'),
('stance_change_high', 'stance', 0.6, 'high'),
('toxicity_spike_medium', 'toxicity', 0.1, 'medium'),
('toxicity_spike_high', 'toxicity', 0.3, 'high'),
('bias_extreme_medium', 'bias', 0.6, 'medium'),
('bias_extreme_high', 'bias', 0.8, 'high')
ON CONFLICT (threshold_name) DO NOTHING;
