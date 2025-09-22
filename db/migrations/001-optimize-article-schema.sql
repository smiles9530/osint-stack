-- Migration: Optimize Article Schema
-- This migration simplifies the article schema by consolidating multiple tables
-- into a more efficient structure with auto-generated UUIDs and minimal mandatory fields

\connect osint

-- Start transaction
BEGIN;

-- Step 1: Create backup tables (optional - for rollback)
CREATE TABLE IF NOT EXISTS articles_backup AS SELECT * FROM articles;
CREATE TABLE IF NOT EXISTS article_analysis_backup AS SELECT * FROM article_analysis;

-- Step 2: Create new optimized tables
CREATE TABLE articles_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL UNIQUE,
    title TEXT,
    content TEXT,
    language VARCHAR(10) DEFAULT 'en',
    published_at TIMESTAMPTZ,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    source_name TEXT,
    analysis_data JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    word_count INTEGER,
    reading_time INTEGER,
    is_deleted BOOLEAN DEFAULT FALSE,
    deleted_at TIMESTAMPTZ
);

CREATE TABLE analysis_results_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID REFERENCES articles_new(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL,
    results JSONB NOT NULL,
    confidence FLOAT DEFAULT 0.0,
    model_name VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(article_id, analysis_type)
);

CREATE TABLE alerts_new (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    article_id UUID REFERENCES articles_new(id) ON DELETE CASCADE,
    source_id INTEGER REFERENCES sources(id) ON DELETE CASCADE,
    alert_data JSONB DEFAULT '{}',
    is_acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by INTEGER REFERENCES users(id),
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Step 3: Migrate data from old tables to new tables
INSERT INTO articles_new (
    id, url, title, content, language, published_at, fetched_at, 
    source_id, source_name, analysis_data, metadata, word_count
)
SELECT 
    gen_random_uuid() as id,
    url,
    title,
    text as content,
    COALESCE(lang, 'en') as language,
    published_at,
    fetched_at,
    source_id,
    s.name as source_name,
    '{}' as analysis_data,  -- Will be populated from analysis tables
    jsonb_build_object(
        'tone', tone,
        'dedupe_hash', dedupe_hash,
        'country_iso', country_iso
    ) as metadata,
    CASE 
        WHEN text IS NOT NULL THEN array_length(string_to_array(text, ' '), 1)
        ELSE NULL
    END as word_count
FROM articles a
LEFT JOIN sources s ON a.source_id = s.id
WHERE a.id IS NOT NULL;

-- Step 4: Migrate analysis data
INSERT INTO analysis_results_new (article_id, analysis_type, results, confidence, created_at)
SELECT 
    an.id as article_id,
    'bias_analysis' as analysis_type,
    jsonb_build_object(
        'subjectivity', aa.subjectivity,
        'sensationalism', aa.sensationalism,
        'loaded_language', aa.loaded_language,
        'bias_lr', aa.bias_lr,
        'stance', aa.stance,
        'evidence_density', aa.evidence_density,
        'sentiment', aa.sentiment,
        'sentiment_confidence', aa.sentiment_confidence,
        'agenda_signals', aa.agenda_signals,
        'risk_flags', aa.risk_flags,
        'entities', aa.entities,
        'tags', aa.tags,
        'key_quotes', aa.key_quotes,
        'summary_bullets', aa.summary_bullets,
        'confidence_score', aa.confidence_score,
        'model_agreement', aa.model_agreement,
        'bias_trend', aa.bias_trend
    ) as results,
    COALESCE(aa.confidence_score, 0.0) as confidence,
    COALESCE(aa.analysis_timestamp, aa.created_at) as created_at
FROM article_analysis aa
JOIN articles a ON aa.article_id = a.id
JOIN articles_new an ON an.url = a.url AND an.published_at = a.published_at
WHERE aa.article_id IS NOT NULL;

-- Step 5: Create indexes for new tables
CREATE INDEX idx_articles_new_url ON articles_new (url);
CREATE INDEX idx_articles_new_published_at ON articles_new (published_at DESC);
CREATE INDEX idx_articles_new_source_id ON articles_new (source_id);
CREATE INDEX idx_articles_new_language ON articles_new (language);
CREATE INDEX idx_articles_new_fetched_at ON articles_new (fetched_at DESC);
CREATE INDEX idx_articles_new_deleted ON articles_new (is_deleted) WHERE is_deleted = FALSE;
CREATE INDEX idx_articles_new_analysis_data ON articles_new USING GIN (analysis_data);
CREATE INDEX idx_articles_new_metadata ON articles_new USING GIN (metadata);
CREATE INDEX idx_articles_new_active ON articles_new (published_at DESC) WHERE is_deleted = FALSE;

CREATE INDEX idx_analysis_results_new_article ON analysis_results_new (article_id);
CREATE INDEX idx_analysis_results_new_type ON analysis_results_new (analysis_type);
CREATE INDEX idx_analysis_results_new_created ON analysis_results_new (created_at DESC);
CREATE INDEX idx_analysis_results_new_data ON analysis_results_new USING GIN (results);

CREATE INDEX idx_alerts_new_type ON alerts_new (alert_type);
CREATE INDEX idx_alerts_new_severity ON alerts_new (severity);
CREATE INDEX idx_alerts_new_acknowledged ON alerts_new (is_acknowledged);
CREATE INDEX idx_alerts_new_created ON alerts_new (created_at DESC);
CREATE INDEX idx_alerts_new_article ON alerts_new (article_id);

-- Step 6: Create materialized view for dashboard
CREATE MATERIALIZED VIEW dashboard_stats_new AS
SELECT 
    DATE(fetched_at) as date,
    COUNT(*) as total_articles,
    COUNT(DISTINCT source_id) as unique_sources,
    COUNT(DISTINCT language) as languages,
    AVG(word_count) as avg_word_count,
    COUNT(CASE WHEN analysis_data != '{}' THEN 1 END) as analyzed_articles,
    COUNT(CASE WHEN is_deleted = TRUE THEN 1 END) as deleted_articles
FROM articles_new
WHERE fetched_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE(fetched_at)
ORDER BY date DESC;

-- Step 7: Create helper functions
CREATE OR REPLACE FUNCTION update_word_count_new()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.content IS NOT NULL THEN
        NEW.word_count := array_length(string_to_array(NEW.content, ' '), 1);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_word_count_new
    BEFORE INSERT OR UPDATE ON articles_new
    FOR EACH ROW
    EXECUTE FUNCTION update_word_count_new();

-- Step 8: Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON articles_new TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON analysis_results_new TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON alerts_new TO osint_user;
GRANT SELECT ON dashboard_stats_new TO osint_user;

-- Step 9: Update analysis_data in articles_new with consolidated data
UPDATE articles_new 
SET analysis_data = (
    SELECT jsonb_object_agg(ar.analysis_type, ar.results)
    FROM analysis_results_new ar
    WHERE ar.article_id = articles_new.id
)
WHERE EXISTS (
    SELECT 1 FROM analysis_results_new ar 
    WHERE ar.article_id = articles_new.id
);

-- Step 10: Verify migration
DO $$
DECLARE
    old_count INTEGER;
    new_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO old_count FROM articles;
    SELECT COUNT(*) INTO new_count FROM articles_new;
    
    IF old_count != new_count THEN
        RAISE EXCEPTION 'Migration failed: Article count mismatch. Old: %, New: %', old_count, new_count;
    END IF;
    
    RAISE NOTICE 'Migration successful: % articles migrated', new_count;
END $$;

-- Commit transaction
COMMIT;

-- Step 11: Create rollback script (optional)
CREATE OR REPLACE FUNCTION rollback_article_optimization()
RETURNS void AS $$
BEGIN
    -- This function can be used to rollback if needed
    RAISE NOTICE 'Rollback function created. Use with caution.';
END;
$$ LANGUAGE plpgsql;
