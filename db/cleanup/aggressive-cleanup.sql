-- Aggressive Database Cleanup Script
-- Removes ALL article-related tables and partitions while preserving users
\connect osint

-- Step 1: Drop all article partitions first
DO $$
DECLARE
    partition_name TEXT;
BEGIN
    -- Drop all article partitions
    FOR partition_name IN 
        SELECT schemaname||'.'||tablename 
        FROM pg_tables 
        WHERE tablename LIKE 'articles_%' 
        AND schemaname = 'public'
    LOOP
        EXECUTE 'DROP TABLE IF EXISTS ' || partition_name || ' CASCADE';
        RAISE NOTICE 'Dropped partition: %', partition_name;
    END LOOP;
END $$;

-- Step 2: Drop the main articles table
DROP TABLE IF EXISTS articles CASCADE;

-- Step 3: Drop all article-related tables
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

-- Step 4: Drop materialized views
DROP MATERIALIZED VIEW IF EXISTS analysis_dashboard_stats CASCADE;
DROP MATERIALIZED VIEW IF EXISTS dashboard_stats CASCADE;
DROP MATERIALIZED VIEW IF EXISTS article_stats CASCADE;

-- Step 5: Drop all functions
DO $$
DECLARE
    func_name TEXT;
BEGIN
    FOR func_name IN 
        SELECT proname||'('||pg_get_function_identity_arguments(oid)||')'
        FROM pg_proc 
        WHERE pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
        AND proname NOT LIKE 'pg_%'
    LOOP
        EXECUTE 'DROP FUNCTION IF EXISTS ' || func_name || ' CASCADE';
    END LOOP;
END $$;

-- Step 6: Drop all sequences
DO $$
DECLARE
    seq_name TEXT;
BEGIN
    FOR seq_name IN 
        SELECT sequencename
        FROM pg_sequences 
        WHERE schemaname = 'public'
        AND sequencename NOT LIKE 'pg_%'
    LOOP
        EXECUTE 'DROP SEQUENCE IF EXISTS ' || seq_name || ' CASCADE';
    END LOOP;
END $$;

-- Step 7: Drop all indexes
DO $$
DECLARE
    idx_name TEXT;
BEGIN
    FOR idx_name IN 
        SELECT indexname
        FROM pg_indexes 
        WHERE schemaname = 'public'
        AND indexname NOT LIKE 'pg_%'
    LOOP
        EXECUTE 'DROP INDEX IF EXISTS ' || idx_name || ' CASCADE';
    END LOOP;
END $$;

-- Step 8: Create a simple, clean articles table
CREATE TABLE articles (
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

-- Step 9: Create basic indexes
CREATE INDEX idx_articles_url ON articles (url);
CREATE INDEX idx_articles_published_at ON articles (published_at DESC);
CREATE INDEX idx_articles_source_id ON articles (source_id);
CREATE INDEX idx_articles_language ON articles (language);
CREATE INDEX idx_articles_fetched_at ON articles (fetched_at DESC);
CREATE INDEX idx_articles_deleted ON articles (is_deleted) WHERE is_deleted = FALSE;
CREATE INDEX idx_articles_metadata ON articles USING GIN (metadata);

-- Step 10: Verify cleanup
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
    
    RAISE NOTICE 'Aggressive cleanup completed successfully. Users table preserved.';
END $$;

-- Final verification
\echo 'Database cleanup completed successfully!'
\echo 'Remaining tables:'
\dt
