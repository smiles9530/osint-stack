-- Database Cleanup Script
-- Removes all complex article tables while preserving users and basic structure
\connect osint

-- Start transaction for safety
BEGIN;

-- Step 1: Drop all article-related tables (in dependency order)
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
DROP TABLE IF EXISTS feedback CASCADE;
DROP TABLE IF EXISTS bandit_state CASCADE;
DROP TABLE IF EXISTS article_topics CASCADE;
DROP TABLE IF EXISTS topics CASCADE;
DROP TABLE IF EXISTS topic_timeseries CASCADE;
DROP TABLE IF EXISTS embeddings CASCADE;
DROP TABLE IF EXISTS reranking_cache CASCADE;
DROP TABLE IF EXISTS topic_clusters CASCADE;
DROP TABLE IF EXISTS article_topic_clusters CASCADE;
DROP TABLE IF EXISTS entity_nodes CASCADE;
DROP TABLE IF EXISTS entity_relationships CASCADE;
DROP TABLE IF EXISTS geographic_entities CASCADE;
DROP TABLE IF EXISTS article_geotags CASCADE;

-- Step 2: Drop the main articles table and its partitions
DROP TABLE IF EXISTS articles CASCADE;

-- Step 3: Drop materialized views
DROP MATERIALIZED VIEW IF EXISTS analysis_dashboard_stats CASCADE;
DROP MATERIALIZED VIEW IF EXISTS dashboard_stats CASCADE;

-- Step 4: Drop functions related to articles
DROP FUNCTION IF EXISTS refresh_analysis_dashboard_stats() CASCADE;
DROP FUNCTION IF EXISTS get_source_analysis_summary(VARCHAR, INTEGER) CASCADE;
DROP FUNCTION IF EXISTS get_topic_analysis_summary(VARCHAR, INTEGER) CASCADE;
DROP FUNCTION IF EXISTS detect_analysis_anomalies(INTEGER) CASCADE;
DROP FUNCTION IF EXISTS refresh_dashboard_stats() CASCADE;
DROP FUNCTION IF EXISTS get_article_with_analysis(UUID) CASCADE;
DROP FUNCTION IF EXISTS search_articles(TEXT, INTEGER, VARCHAR, INTEGER, INTEGER) CASCADE;
DROP FUNCTION IF EXISTS update_word_count() CASCADE;
DROP FUNCTION IF EXISTS update_word_count_new() CASCADE;
DROP FUNCTION IF EXISTS rollback_article_optimization() CASCADE;

-- Step 5: Drop triggers
DROP TRIGGER IF EXISTS trigger_update_word_count ON articles CASCADE;
DROP TRIGGER IF EXISTS trigger_update_word_count_new ON articles_new CASCADE;

-- Step 6: Drop indexes that might be left behind
DROP INDEX IF EXISTS idx_articles_published_at CASCADE;
DROP INDEX IF EXISTS idx_articles_country CASCADE;
DROP INDEX IF EXISTS idx_articles_source_id CASCADE;
DROP INDEX IF EXISTS idx_articles_lang CASCADE;
DROP INDEX IF EXISTS idx_articles_url CASCADE;
DROP INDEX IF EXISTS idx_articles_fetched_at CASCADE;
DROP INDEX IF EXISTS idx_articles_deleted CASCADE;
DROP INDEX IF EXISTS idx_articles_analysis_data CASCADE;
DROP INDEX IF EXISTS idx_articles_metadata CASCADE;
DROP INDEX IF EXISTS idx_articles_active CASCADE;
DROP INDEX IF EXISTS idx_analysis_results_article CASCADE;
DROP INDEX IF EXISTS idx_analysis_results_type CASCADE;
DROP INDEX IF EXISTS idx_analysis_results_created CASCADE;
DROP INDEX IF EXISTS idx_analysis_results_data CASCADE;
DROP INDEX IF EXISTS idx_alerts_type CASCADE;
DROP INDEX IF EXISTS idx_alerts_severity CASCADE;
DROP INDEX IF EXISTS idx_alerts_acknowledged CASCADE;
DROP INDEX IF EXISTS idx_alerts_created CASCADE;
DROP INDEX IF EXISTS idx_alerts_article CASCADE;

-- Step 7: Drop any remaining article-related tables
DROP TABLE IF EXISTS articles_new CASCADE;
DROP TABLE IF EXISTS analysis_results CASCADE;
DROP TABLE IF EXISTS analysis_results_new CASCADE;
DROP TABLE IF EXISTS alerts CASCADE;
DROP TABLE IF EXISTS alerts_new CASCADE;

-- Step 8: Drop TimescaleDB hypertables (if they exist)
-- Note: This might fail if TimescaleDB is not installed, that's okay
DO $$
BEGIN
    -- Try to drop hypertables, ignore if they don't exist
    BEGIN
        PERFORM drop_hypertable('topic_timeseries', if_exists => TRUE);
    EXCEPTION
        WHEN undefined_function THEN
            -- TimescaleDB not installed, continue
            NULL;
    END;
END $$;

-- Step 9: Clean up any remaining sequences
DROP SEQUENCE IF EXISTS articles_id_seq CASCADE;
DROP SEQUENCE IF EXISTS article_analysis_id_seq CASCADE;
DROP SEQUENCE IF EXISTS article_features_id_seq CASCADE;
DROP SEQUENCE IF EXISTS article_sentiment_id_seq CASCADE;
DROP SEQUENCE IF EXISTS article_entities_id_seq CASCADE;
DROP SEQUENCE IF EXISTS article_topics_ml_id_seq CASCADE;
DROP SEQUENCE IF EXISTS article_anomalies_id_seq CASCADE;
DROP SEQUENCE IF EXISTS article_geotags_id_seq CASCADE;
DROP SEQUENCE IF EXISTS media_files_id_seq CASCADE;
DROP SEQUENCE IF EXISTS stance_sentiment_analysis_id_seq CASCADE;
DROP SEQUENCE IF EXISTS aggregated_analysis_id_seq CASCADE;
DROP SEQUENCE IF EXISTS source_analysis_daily_id_seq CASCADE;
DROP SEQUENCE IF EXISTS topic_analysis_daily_id_seq CASCADE;
DROP SEQUENCE IF EXISTS analysis_alerts_id_seq CASCADE;
DROP SEQUENCE IF EXISTS analysis_thresholds_id_seq CASCADE;
DROP SEQUENCE IF EXISTS feedback_id_seq CASCADE;
DROP SEQUENCE IF EXISTS bandit_state_id_seq CASCADE;
DROP SEQUENCE IF EXISTS article_topics_id_seq CASCADE;
DROP SEQUENCE IF EXISTS topics_id_seq CASCADE;
DROP SEQUENCE IF EXISTS topic_timeseries_id_seq CASCADE;
DROP SEQUENCE IF EXISTS embeddings_id_seq CASCADE;
DROP SEQUENCE IF EXISTS reranking_cache_id_seq CASCADE;
DROP SEQUENCE IF EXISTS topic_clusters_id_seq CASCADE;
DROP SEQUENCE IF EXISTS article_topic_clusters_id_seq CASCADE;
DROP SEQUENCE IF EXISTS entity_nodes_id_seq CASCADE;
DROP SEQUENCE IF EXISTS entity_relationships_id_seq CASCADE;
DROP SEQUENCE IF EXISTS geographic_entities_id_seq CASCADE;

-- Step 10: Verify cleanup - show remaining tables
DO $$
DECLARE
    table_count INTEGER;
    table_list TEXT;
BEGIN
    SELECT COUNT(*), string_agg(tablename, ', ')
    INTO table_count, table_list
    FROM pg_tables 
    WHERE schemaname = 'public' 
    AND tablename NOT LIKE 'pg_%'
    AND tablename NOT LIKE 'sql_%';
    
    RAISE NOTICE 'Remaining tables (%): %', table_count, table_list;
    
    -- Ensure users table still exists
    IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'users' AND schemaname = 'public') THEN
        RAISE EXCEPTION 'ERROR: Users table was accidentally dropped!';
    END IF;
    
    RAISE NOTICE 'Cleanup completed successfully. Users table preserved.';
END $$;

-- Step 11: Create a simple, clean articles table for future use
CREATE TABLE IF NOT EXISTS articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT NOT NULL UNIQUE,
    title TEXT,
    content TEXT,
    language VARCHAR(10) DEFAULT 'en',
    published_at TIMESTAMPTZ,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    source_name TEXT,
    metadata JSONB DEFAULT '{}',
    is_deleted BOOLEAN DEFAULT FALSE,
    deleted_at TIMESTAMPTZ
);

-- Create basic indexes
CREATE INDEX IF NOT EXISTS idx_articles_url ON articles (url);
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_source_id ON articles (source_id);
CREATE INDEX IF NOT EXISTS idx_articles_language ON articles (language);
CREATE INDEX IF NOT EXISTS idx_articles_fetched_at ON articles (fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_deleted ON articles (is_deleted) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_articles_metadata ON articles USING GIN (metadata);

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON articles TO osint_user;

-- Commit transaction
COMMIT;

-- Final verification
\echo 'Database cleanup completed successfully!'
\echo 'Remaining tables:'
\dt
