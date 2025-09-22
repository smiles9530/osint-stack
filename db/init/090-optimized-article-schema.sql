-- Optimized Article Schema - Simplified and Streamlined
\connect osint

-- Drop existing complex tables (in correct order due to dependencies)
DROP TABLE IF EXISTS article_analysis CASCADE;
DROP TABLE IF EXISTS article_features CASCADE;
DROP TABLE IF EXISTS article_sentiment CASCADE;
DROP TABLE IF EXISTS article_entities CASCADE;
DROP TABLE IF EXISTS article_topics_ml CASCADE;
DROP TABLE IF EXISTS article_anomalies CASCADE;
DROP TABLE IF EXISTS article_geotags CASCADE;
DROP TABLE IF EXISTS media_files CASCADE;
DROP TABLE IF EXISTS stance_sentiment_analysis CASCADE;
DROP TABLE IF EXISTS aggregated_analysis CASCADE;
DROP TABLE IF EXISTS source_analysis_daily CASCADE;
DROP TABLE IF EXISTS topic_analysis_daily CASCADE;
DROP TABLE IF EXISTS analysis_alerts CASCADE;
DROP TABLE IF EXISTS analysis_thresholds CASCADE;

-- Drop and recreate articles table with optimized schema
DROP TABLE IF EXISTS articles CASCADE;

-- Optimized articles table - simplified structure
CREATE TABLE articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),  -- Auto-generated UUID
    url TEXT NOT NULL UNIQUE,                       -- Only mandatory field
    title TEXT,                                     -- Optional
    content TEXT,                                   -- Optional (renamed from 'text')
    language VARCHAR(10) DEFAULT 'en',              -- Optional with default
    published_at TIMESTAMPTZ,                       -- Optional
    fetched_at TIMESTAMPTZ DEFAULT NOW(),           -- Auto-generated
    source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,  -- Optional
    source_name TEXT,                               -- Optional (denormalized for performance)
    
    -- Analysis data (JSONB for flexibility)
    analysis_data JSONB DEFAULT '{}',               -- All analysis results in one field
    metadata JSONB DEFAULT '{}',                    -- Additional metadata
    
    -- Performance fields (optional)
    word_count INTEGER,
    reading_time INTEGER,                           -- in minutes
    
    -- Soft delete
    is_deleted BOOLEAN DEFAULT FALSE,
    deleted_at TIMESTAMPTZ
);

-- Create optimized indexes
CREATE INDEX idx_articles_url ON articles (url);
CREATE INDEX idx_articles_published_at ON articles (published_at DESC);
CREATE INDEX idx_articles_source_id ON articles (source_id);
CREATE INDEX idx_articles_language ON articles (language);
CREATE INDEX idx_articles_fetched_at ON articles (fetched_at DESC);
CREATE INDEX idx_articles_deleted ON articles (is_deleted) WHERE is_deleted = FALSE;

-- GIN indexes for JSONB fields (for fast JSON queries)
CREATE INDEX idx_articles_analysis_data ON articles USING GIN (analysis_data);
CREATE INDEX idx_articles_metadata ON articles USING GIN (metadata);

-- Partial index for active articles only
CREATE INDEX idx_articles_active ON articles (published_at DESC) WHERE is_deleted = FALSE;

-- Create simplified analysis results table
CREATE TABLE analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL,  -- 'sentiment', 'bias', 'stance', 'toxicity', etc.
    results JSONB NOT NULL,              -- Flexible results storage
    confidence FLOAT DEFAULT 0.0,
    model_name VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(article_id, analysis_type)  -- One analysis per type per article
);

-- Indexes for analysis results
CREATE INDEX idx_analysis_results_article ON analysis_results (article_id);
CREATE INDEX idx_analysis_results_type ON analysis_results (analysis_type);
CREATE INDEX idx_analysis_results_created ON analysis_results (created_at DESC);
CREATE INDEX idx_analysis_results_data ON analysis_results USING GIN (results);

-- Create simplified alerts table
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    source_id INTEGER REFERENCES sources(id) ON DELETE CASCADE,
    alert_data JSONB DEFAULT '{}',
    is_acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by INTEGER REFERENCES users(id),
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for alerts
CREATE INDEX idx_alerts_type ON alerts (alert_type);
CREATE INDEX idx_alerts_severity ON alerts (severity);
CREATE INDEX idx_alerts_acknowledged ON alerts (is_acknowledged);
CREATE INDEX idx_alerts_created ON alerts (created_at DESC);
CREATE INDEX idx_alerts_article ON alerts (article_id);

-- Create materialized view for dashboard (simplified)
CREATE MATERIALIZED VIEW dashboard_stats AS
SELECT 
    DATE(fetched_at) as date,
    COUNT(*) as total_articles,
    COUNT(DISTINCT source_id) as unique_sources,
    COUNT(DISTINCT language) as languages,
    AVG(word_count) as avg_word_count,
    COUNT(CASE WHEN analysis_data != '{}' THEN 1 END) as analyzed_articles,
    COUNT(CASE WHEN is_deleted = TRUE THEN 1 END) as deleted_articles
FROM articles
WHERE fetched_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE(fetched_at)
ORDER BY date DESC;

-- Create function to refresh dashboard stats
CREATE OR REPLACE FUNCTION refresh_dashboard_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY dashboard_stats;
END;
$$ LANGUAGE plpgsql;

-- Create function to get article with analysis
CREATE OR REPLACE FUNCTION get_article_with_analysis(p_article_id UUID)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'article', row_to_json(a.*),
        'analysis', (
            SELECT json_object_agg(ar.analysis_type, ar.results)
            FROM analysis_results ar
            WHERE ar.article_id = p_article_id
        )
    ) INTO result
    FROM articles a
    WHERE a.id = p_article_id AND a.is_deleted = FALSE;
    
    RETURN COALESCE(result, '{}'::json);
END;
$$ LANGUAGE plpgsql;

-- Create function to search articles with analysis
CREATE OR REPLACE FUNCTION search_articles(
    p_query TEXT DEFAULT '',
    p_source_id INTEGER DEFAULT NULL,
    p_language VARCHAR(10) DEFAULT NULL,
    p_limit INTEGER DEFAULT 50,
    p_offset INTEGER DEFAULT 0
)
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'articles', json_agg(
            json_build_object(
                'id', a.id,
                'url', a.url,
                'title', a.title,
                'content', a.content,
                'language', a.language,
                'published_at', a.published_at,
                'source_name', a.source_name,
                'word_count', a.word_count,
                'analysis_data', a.analysis_data,
                'metadata', a.metadata
            )
        ),
        'total', COUNT(*),
        'limit', p_limit,
        'offset', p_offset
    ) INTO result
    FROM articles a
    WHERE a.is_deleted = FALSE
    AND (p_query = '' OR a.title ILIKE '%' || p_query || '%' OR a.content ILIKE '%' || p_query || '%')
    AND (p_source_id IS NULL OR a.source_id = p_source_id)
    AND (p_language IS NULL OR a.language = p_language)
    ORDER BY a.published_at DESC
    LIMIT p_limit OFFSET p_offset;
    
    RETURN COALESCE(result, '{"articles": [], "total": 0, "limit": 0, "offset": 0}'::json);
END;
$$ LANGUAGE plpgsql;

-- Insert sample data for testing
INSERT INTO articles (url, title, content, language, published_at, source_name, word_count) VALUES
('https://example.com/news1', 'Sample News Article 1', 'This is sample content for testing the optimized schema.', 'en', NOW() - INTERVAL '1 day', 'Example News', 25),
('https://example.com/news2', 'Sample News Article 2', 'Another sample article for testing purposes.', 'en', NOW() - INTERVAL '2 days', 'Test News', 30);

-- Create trigger to update word_count automatically
CREATE OR REPLACE FUNCTION update_word_count()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.content IS NOT NULL THEN
        NEW.word_count := array_length(string_to_array(NEW.content, ' '), 1);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_word_count
    BEFORE INSERT OR UPDATE ON articles
    FOR EACH ROW
    EXECUTE FUNCTION update_word_count();

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON articles TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON analysis_results TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON alerts TO osint_user;
GRANT SELECT ON dashboard_stats TO osint_user;
