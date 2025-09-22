-- Emergency fix for n8n workflow Reddit ID issue
\connect osint

-- Create a temporary table to handle Reddit IDs before they hit the main articles table
CREATE TABLE IF NOT EXISTS articles_staging (
    id TEXT,  -- Allow any text for staging
    url TEXT NOT NULL,
    title TEXT,
    content TEXT,
    language VARCHAR(10) DEFAULT 'en',
    published_at TIMESTAMPTZ,
    source_name TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create a function to process staging data and insert into main table
CREATE OR REPLACE FUNCTION process_staging_articles()
RETURNS INTEGER AS $$
DECLARE
    staging_record RECORD;
    processed_count INTEGER := 0;
    final_uuid UUID;
    reddit_mapping JSONB;
BEGIN
    -- Process all staging records
    FOR staging_record IN 
        SELECT * FROM articles_staging
        ORDER BY created_at
    LOOP
        -- Handle the ID field
        IF staging_record.id IS NOT NULL THEN
            -- Check if it's a Reddit ID
            IF staging_record.id ~ '^t[123]_[a-zA-Z0-9]+$' THEN
                -- Convert Reddit ID to UUID
                final_uuid := reddit_id_to_uuid(staging_record.id);
                
                -- Create Reddit mapping
                reddit_mapping := jsonb_build_object(
                    'original_reddit_id', staging_record.id,
                    'reddit_type', CASE 
                        WHEN staging_record.id ~ '^t3_' THEN 'post'
                        WHEN staging_record.id ~ '^t1_' THEN 'comment'
                        WHEN staging_record.id ~ '^t2_' THEN 'user'
                        ELSE 'unknown'
                    END,
                    'mapped_at', NOW()
                );
                
                -- Merge with existing metadata
                staging_record.metadata := COALESCE(staging_record.metadata, '{}'::jsonb) || jsonb_build_object('reddit_id_mapping', reddit_mapping);
            ELSE
                -- Try to use as UUID directly
                BEGIN
                    final_uuid := staging_record.id::UUID;
                EXCEPTION WHEN invalid_text_representation THEN
                    final_uuid := gen_random_uuid();
                END;
            END IF;
        ELSE
            final_uuid := gen_random_uuid();
        END IF;
        
        -- Insert into main articles table
        INSERT INTO articles (id, url, title, content, language, published_at, source_name, metadata)
        VALUES (final_uuid, staging_record.url, staging_record.title, staging_record.content, 
                staging_record.language, staging_record.published_at, staging_record.source_name, staging_record.metadata)
        ON CONFLICT (url) DO UPDATE SET
            title = COALESCE(EXCLUDED.title, articles.title),
            content = COALESCE(EXCLUDED.content, articles.content),
            language = COALESCE(EXCLUDED.language, articles.language),
            published_at = COALESCE(EXCLUDED.published_at, articles.published_at),
            source_name = COALESCE(EXCLUDED.source_name, articles.source_name),
            metadata = articles.metadata || EXCLUDED.metadata,
            fetched_at = NOW();
        
        processed_count := processed_count + 1;
    END LOOP;
    
    -- Clear processed staging records
    DELETE FROM articles_staging;
    
    RETURN processed_count;
END;
$$ LANGUAGE plpgsql;

-- Create a trigger to automatically process staging data
CREATE OR REPLACE FUNCTION auto_process_staging()
RETURNS TRIGGER AS $$
BEGIN
    -- Process staging data every time a new record is added
    PERFORM process_staging_articles();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger on staging table
DROP TRIGGER IF EXISTS auto_process_staging_trigger ON articles_staging;
CREATE TRIGGER auto_process_staging_trigger
    AFTER INSERT ON articles_staging
    FOR EACH ROW
    EXECUTE FUNCTION auto_process_staging();

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON articles_staging TO osint;
GRANT EXECUTE ON FUNCTION process_staging_articles TO osint;
GRANT EXECUTE ON FUNCTION auto_process_staging TO osint;

-- Test the staging approach
INSERT INTO articles_staging (id, url, title, content, language, published_at, source_name, metadata)
VALUES ('t3_1nm0pvw', 'https://reddit.com/test-staging', 'Test Staging Reddit Post', 'Test content', 'en', NOW(), 'Reddit', '{}');

-- Check if it was processed
SELECT 'Staging test completed!' as status;
SELECT COUNT(*) as staging_records FROM articles_staging;
SELECT COUNT(*) as processed_articles FROM articles WHERE url = 'https://reddit.com/test-staging';
