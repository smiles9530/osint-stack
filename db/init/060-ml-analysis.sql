-- ML Analysis Tables
-- Advanced ML analysis results storage

-- Topic modeling results
CREATE TABLE IF NOT EXISTS topic_analysis (
    id SERIAL PRIMARY KEY,
    analysis_id UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    method VARCHAR(50) NOT NULL, -- 'lda', 'bertopic'
    num_topics INTEGER NOT NULL,
    total_documents INTEGER NOT NULL,
    analysis_time FLOAT NOT NULL,
    parameters JSONB,
    created_by INTEGER REFERENCES users(id)
);

-- Individual topics from analysis
CREATE TABLE IF NOT EXISTS topics (
    id SERIAL PRIMARY KEY,
    analysis_id UUID REFERENCES topic_analysis(analysis_id) ON DELETE CASCADE,
    topic_id INTEGER NOT NULL,
    keywords TEXT[] NOT NULL,
    weight FLOAT NOT NULL,
    coherence_score FLOAT NOT NULL,
    document_count INTEGER NOT NULL,
    document_percentage FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Entity extraction results
CREATE TABLE IF NOT EXISTS entity_extractions (
    id SERIAL PRIMARY KEY,
    article_id INTEGER,
    text TEXT NOT NULL,
    label VARCHAR(100) NOT NULL,
    start_pos INTEGER NOT NULL,
    end_pos INTEGER NOT NULL,
    confidence FLOAT NOT NULL,
    description TEXT,
    extraction_method VARCHAR(50) DEFAULT 'spacy',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by INTEGER REFERENCES users(id),
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);

-- Trend analysis results
CREATE TABLE IF NOT EXISTS trend_analysis (
    id SERIAL PRIMARY KEY,
    analysis_id UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source_ids INTEGER[],
    date_from TIMESTAMP WITH TIME ZONE,
    date_to TIMESTAMP WITH TIME ZONE,
    period VARCHAR(20) NOT NULL, -- 'daily', 'weekly', 'monthly'
    value_type VARCHAR(20) NOT NULL, -- 'count', 'sentiment', 'engagement'
    trend_direction VARCHAR(20) NOT NULL, -- 'increasing', 'decreasing', 'stable'
    trend_strength FLOAT NOT NULL,
    change_rate FLOAT NOT NULL,
    significance FLOAT NOT NULL,
    forecast JSONB,
    data_points INTEGER NOT NULL,
    analysis_time FLOAT NOT NULL,
    created_by INTEGER REFERENCES users(id)
);

-- Anomaly detection results
CREATE TABLE IF NOT EXISTS anomaly_detections (
    id SERIAL PRIMARY KEY,
    analysis_id UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source_ids INTEGER[],
    date_from TIMESTAMP WITH TIME ZONE,
    date_to TIMESTAMP WITH TIME ZONE,
    contamination_rate FLOAT NOT NULL,
    total_anomalies INTEGER NOT NULL,
    anomalies JSONB NOT NULL,
    analysis_time FLOAT NOT NULL,
    created_by INTEGER REFERENCES users(id)
);

-- ML model performance tracking
CREATE TABLE IF NOT EXISTS ml_model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    task_type VARCHAR(50) NOT NULL, -- 'topic_modeling', 'entity_extraction', 'trend_analysis'
    accuracy FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    processing_time FLOAT NOT NULL,
    data_size INTEGER NOT NULL,
    parameters JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ML insights cache
CREATE TABLE IF NOT EXISTS ml_insights_cache (
    id SERIAL PRIMARY KEY,
    cache_key VARCHAR(255) UNIQUE NOT NULL,
    insights_data JSONB NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_topic_analysis_created_at ON topic_analysis(created_at);
CREATE INDEX IF NOT EXISTS idx_topic_analysis_method ON topic_analysis(method);
CREATE INDEX IF NOT EXISTS idx_topics_analysis_id ON topics(analysis_id);
CREATE INDEX IF NOT EXISTS idx_topics_topic_id ON topics(topic_id);

CREATE INDEX IF NOT EXISTS idx_entity_extractions_article_id ON entity_extractions(article_id);
CREATE INDEX IF NOT EXISTS idx_entity_extractions_label ON entity_extractions(label);
CREATE INDEX IF NOT EXISTS idx_entity_extractions_created_at ON entity_extractions(created_at);

CREATE INDEX IF NOT EXISTS idx_trend_analysis_created_at ON trend_analysis(created_at);
CREATE INDEX IF NOT EXISTS idx_trend_analysis_period ON trend_analysis(period);
CREATE INDEX IF NOT EXISTS idx_trend_analysis_value_type ON trend_analysis(value_type);

CREATE INDEX IF NOT EXISTS idx_anomaly_detections_created_at ON anomaly_detections(created_at);
CREATE INDEX IF NOT EXISTS idx_anomaly_detections_contamination ON anomaly_detections(contamination_rate);

CREATE INDEX IF NOT EXISTS idx_ml_model_performance_model_name ON ml_model_performance(model_name);
CREATE INDEX IF NOT EXISTS idx_ml_model_performance_task_type ON ml_model_performance(task_type);
CREATE INDEX IF NOT EXISTS idx_ml_model_performance_created_at ON ml_model_performance(created_at);

CREATE INDEX IF NOT EXISTS idx_ml_insights_cache_key ON ml_insights_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_ml_insights_cache_expires ON ml_insights_cache(expires_at);

-- Partitioning for large tables (if needed)
-- ALTER TABLE entity_extractions SET (timescaledb.compress, timescaledb.compress_segmentby = 'article_id');
-- SELECT create_hypertable('entity_extractions', 'created_at', if_not_exists => TRUE);

-- Functions for ML analysis
CREATE OR REPLACE FUNCTION cleanup_expired_ml_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM ml_insights_cache WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get topic analysis summary
CREATE OR REPLACE FUNCTION get_topic_analysis_summary(analysis_uuid UUID)
RETURNS TABLE (
    topic_id INTEGER,
    keywords TEXT[],
    weight FLOAT,
    coherence_score FLOAT,
    document_count INTEGER,
    document_percentage FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.topic_id,
        t.keywords,
        t.weight,
        t.coherence_score,
        t.document_count,
        t.document_percentage
    FROM topics t
    WHERE t.analysis_id = analysis_uuid
    ORDER BY t.weight DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to get entity statistics
CREATE OR REPLACE FUNCTION get_entity_statistics(
    start_date TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    end_date TIMESTAMP WITH TIME ZONE DEFAULT NULL
)
RETURNS TABLE (
    label VARCHAR(100),
    count BIGINT,
    avg_confidence FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ee.label,
        COUNT(*) as count,
        AVG(ee.confidence) as avg_confidence
    FROM entity_extractions ee
    WHERE (start_date IS NULL OR ee.created_at >= start_date)
      AND (end_date IS NULL OR ee.created_at <= end_date)
    GROUP BY ee.label
    ORDER BY count DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to get trend analysis history
CREATE OR REPLACE FUNCTION get_trend_history(
    period_type VARCHAR(20) DEFAULT 'daily',
    days_back INTEGER DEFAULT 30
)
RETURNS TABLE (
    analysis_date TIMESTAMP WITH TIME ZONE,
    trend_direction VARCHAR(20),
    trend_strength FLOAT,
    change_rate FLOAT,
    significance FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ta.created_at as analysis_date,
        ta.trend_direction,
        ta.trend_strength,
        ta.change_rate,
        ta.significance
    FROM trend_analysis ta
    WHERE ta.period = period_type
      AND ta.created_at >= NOW() - INTERVAL '1 day' * days_back
    ORDER BY ta.created_at DESC;
END;
$$ LANGUAGE plpgsql;

-- Insert sample ML model performance data
INSERT INTO ml_model_performance (model_name, model_version, task_type, accuracy, processing_time, data_size, parameters)
VALUES 
    ('spacy-ner', '3.7.2', 'entity_extraction', 0.89, 0.5, 1000, '{"model": "en_core_web_sm"}'),
    ('lda-topic', '1.3.2', 'topic_modeling', 0.75, 2.3, 500, '{"num_topics": 10, "max_iter": 100}'),
    ('trend-analysis', '0.14.0', 'trend_analysis', 0.82, 1.1, 100, '{"period": "daily", "method": "linear_regression"}')
ON CONFLICT DO NOTHING;

-- Create a view for ML analysis dashboard
CREATE OR REPLACE VIEW ml_analysis_dashboard AS
SELECT 
    'topic_analysis' as analysis_type,
    COUNT(*) as total_analyses,
    AVG(analysis_time) as avg_analysis_time,
    MAX(created_at) as last_analysis
FROM topic_analysis
UNION ALL
SELECT 
    'entity_extraction' as analysis_type,
    COUNT(*) as total_analyses,
    0.0 as avg_analysis_time,
    MAX(created_at) as last_analysis
FROM entity_extractions
UNION ALL
SELECT 
    'trend_analysis' as analysis_type,
    COUNT(*) as total_analyses,
    AVG(analysis_time) as avg_analysis_time,
    MAX(created_at) as last_analysis
FROM trend_analysis
UNION ALL
SELECT 
    'anomaly_detection' as analysis_type,
    COUNT(*) as total_analyses,
    AVG(analysis_time) as avg_analysis_time,
    MAX(created_at) as last_analysis
FROM anomaly_detections;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO osint;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO osint;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO osint;
