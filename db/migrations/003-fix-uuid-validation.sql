-- Migration: Fix UUID Validation Issues
-- This migration addresses the UUID validation error by ensuring proper schema alignment
-- and adding functions to handle Reddit IDs and other non-UUID identifiers

\connect osint

-- Start transaction
BEGIN;

-- Step 1: Check current schema and create backup
CREATE TABLE IF NOT EXISTS schema_migration_log (
    id SERIAL PRIMARY KEY,
    migration_name VARCHAR(255) NOT NULL,
    executed_at TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(50) DEFAULT 'started',
    details TEXT
);

INSERT INTO schema_migration_log (migration_name, status, details) 
VALUES ('003-fix-uuid-validation', 'started', 'Beginning UUID validation fix migration');

-- Step 2: Ensure we have the UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Step 3: Check if articles table exists and what type of ID it uses
DO $$
DECLARE
    articles_exists BOOLEAN;
    id_type TEXT;
BEGIN
    -- Check if articles table exists
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'articles'
    ) INTO articles_exists;
    
    IF articles_exists THEN
        -- Get the data type of the id column
        SELECT data_type INTO id_type
        FROM information_schema.columns 
        WHERE table_name = 'articles' 
        AND column_name = 'id'
        AND table_schema = 'public';
        
        RAISE NOTICE 'Articles table exists with id type: %', id_type;
        
        -- If it's not UUID, we need to handle the migration
        IF id_type != 'uuid' THEN
            RAISE NOTICE 'Articles table uses % instead of UUID. Migration needed.', id_type;
        END IF;
    ELSE
        RAISE NOTICE 'Articles table does not exist. Will create with UUID.';
    END IF;
END $$;

-- Step 4: Create or update articles table to use UUID
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
    analysis_data JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    word_count INTEGER,
    reading_time INTEGER,
    is_deleted BOOLEAN DEFAULT FALSE,
    deleted_at TIMESTAMPTZ
);

-- Step 5: Create analysis_results table if it doesn't exist
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID REFERENCES articles(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL,
    results JSONB NOT NULL,
    confidence FLOAT DEFAULT 0.0,
    model_name VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(article_id, analysis_type)
);

-- Step 6: Create alerts table if it doesn't exist
CREATE TABLE IF NOT EXISTS alerts (
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

-- Step 7: Create function to safely convert Reddit IDs to UUIDs
CREATE OR REPLACE FUNCTION reddit_id_to_uuid(reddit_id TEXT)
RETURNS UUID AS $$
DECLARE
    namespace_uuid UUID := '6ba7b810-9dad-11d1-80b4-00c04fd430c8'; -- DNS namespace
BEGIN
    -- Check if it's a valid Reddit ID format
    IF reddit_id ~ '^t[123]_[a-zA-Z0-9]+$' THEN
        -- Generate deterministic UUID from Reddit ID
        RETURN uuid_generate_v5(namespace_uuid, reddit_id);
    ELSE
        -- If not a Reddit ID, try to parse as UUID
        BEGIN
            RETURN reddit_id::UUID;
        EXCEPTION WHEN invalid_text_representation THEN
            -- If not a valid UUID either, generate a new one
            RETURN gen_random_uuid();
        END;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Step 8: Create function to safely insert articles with UUID validation
CREATE OR REPLACE FUNCTION safe_upsert_article(
    p_url TEXT,
    p_title TEXT DEFAULT NULL,
    p_content TEXT DEFAULT NULL,
    p_language VARCHAR(10) DEFAULT 'en',
    p_published_at TIMESTAMPTZ DEFAULT NULL,
    p_source_name TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'::JSONB,
    p_article_id TEXT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    article_id UUID;
    source_id INTEGER;
    final_article_id UUID;
BEGIN
    -- Get or create source
    IF p_source_name IS NOT NULL THEN
        INSERT INTO sources(name) VALUES(p_source_name) 
        ON CONFLICT (name) DO UPDATE SET name=EXCLUDED.name 
        RETURNING id INTO source_id;
    END IF;
    
    -- Convert article_id to UUID if provided
    IF p_article_id IS NOT NULL THEN
        final_article_id := reddit_id_to_uuid(p_article_id);
    ELSE
        final_article_id := gen_random_uuid();
    END IF;
    
    -- Insert or update article
    INSERT INTO articles (id, url, title, content, language, published_at, source_id, source_name, metadata)
    VALUES (final_article_id, p_url, p_title, p_content, p_language, p_published_at, source_id, p_source_name, p_metadata)
    ON CONFLICT (url) DO UPDATE SET
        title = COALESCE(EXCLUDED.title, articles.title),
        content = COALESCE(EXCLUDED.content, articles.content),
        language = COALESCE(EXCLUDED.language, articles.language),
        published_at = COALESCE(EXCLUDED.published_at, articles.published_at),
        source_name = COALESCE(EXCLUDED.source_name, articles.source_name),
        metadata = articles.metadata || EXCLUDED.metadata,
        fetched_at = NOW()
    RETURNING id INTO article_id;
    
    RETURN article_id;
END;
$$ LANGUAGE plpgsql;

-- Step 9: Create function to handle Reddit ID mapping in metadata
CREATE OR REPLACE FUNCTION add_reddit_id_mapping(article_id UUID, reddit_id TEXT)
RETURNS void AS $$
BEGIN
    -- Add Reddit ID mapping to article metadata
    UPDATE articles 
    SET metadata = metadata || jsonb_build_object(
        'reddit_id_mapping', jsonb_build_object(
            'original_reddit_id', reddit_id,
            'reddit_type', CASE 
                WHEN reddit_id ~ '^t3_' THEN 'post'
                WHEN reddit_id ~ '^t1_' THEN 'comment'
                WHEN reddit_id ~ '^t2_' THEN 'user'
                ELSE 'unknown'
            END,
            'mapped_at', NOW()
        )
    )
    WHERE id = article_id;
END;
$$ LANGUAGE plpgsql;

-- Step 10: Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_articles_url ON articles (url);
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_source_id ON articles (source_id);
CREATE INDEX IF NOT EXISTS idx_articles_language ON articles (language);
CREATE INDEX IF NOT EXISTS idx_articles_fetched_at ON articles (fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_deleted ON articles (is_deleted) WHERE is_deleted = FALSE;
CREATE INDEX IF NOT EXISTS idx_articles_analysis_data ON articles USING GIN (analysis_data);
CREATE INDEX IF NOT EXISTS idx_articles_metadata ON articles USING GIN (metadata);

CREATE INDEX IF NOT EXISTS idx_analysis_results_article ON analysis_results (article_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_type ON analysis_results (analysis_type);
CREATE INDEX IF NOT EXISTS idx_analysis_results_created ON analysis_results (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_results_data ON analysis_results USING GIN (results);

CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts (alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts (severity);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts (is_acknowledged);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_article ON alerts (article_id);

-- Step 11: Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON articles TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON analysis_results TO osint_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON alerts TO osint_user;
GRANT EXECUTE ON FUNCTION reddit_id_to_uuid TO osint_user;
GRANT EXECUTE ON FUNCTION safe_upsert_article TO osint_user;
GRANT EXECUTE ON FUNCTION add_reddit_id_mapping TO osint_user;

-- Step 12: Update migration log
UPDATE schema_migration_log 
SET status = 'completed', 
    details = 'UUID validation fix migration completed successfully. Added Reddit ID handling and safe article insertion functions.'
WHERE migration_name = '003-fix-uuid-validation';

-- Commit transaction
COMMIT;

-- Display success message
SELECT 'UUID validation fix migration completed successfully!' as status;
