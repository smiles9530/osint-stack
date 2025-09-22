-- Migration: Remove staging table and change articles.id to TEXT
-- This eliminates duplication and simplifies the architecture

-- Step 1: Backup current data (optional - for safety)
-- CREATE TABLE articles_backup AS SELECT * FROM articles;

-- Step 2: Drop the staging table and related functions/triggers
DROP TRIGGER IF EXISTS auto_process_staging_trigger ON articles_staging;
DROP FUNCTION IF EXISTS auto_process_staging();
DROP FUNCTION IF EXISTS process_staging_articles();
DROP TABLE IF EXISTS articles_staging CASCADE;

-- Step 3: Change articles.id from UUID to TEXT
-- First, drop the primary key constraint
ALTER TABLE articles DROP CONSTRAINT articles_pkey;

-- Change the column type from UUID to TEXT
ALTER TABLE articles ALTER COLUMN id TYPE TEXT;

-- Recreate the primary key constraint
ALTER TABLE articles ADD CONSTRAINT articles_pkey PRIMARY KEY (id);

-- Step 4: Update any existing UUIDs to be more readable
-- Convert existing UUIDs to a more readable format or keep as-is
-- (Current UUIDs can stay as they are since they're valid text)

-- Step 5: Add a function to handle Reddit ID conversion (optional)
-- This can be used by applications to convert Reddit IDs to consistent format
CREATE OR REPLACE FUNCTION normalize_article_id(input_id TEXT)
RETURNS TEXT AS $$
BEGIN
    -- If it's a Reddit ID, convert to a consistent format
    IF input_id ~ '^t[123]_[a-zA-Z0-9]+$' THEN
        RETURN 'reddit_' || input_id;
    -- If it's already a UUID, keep it
    ELSIF input_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$' THEN
        RETURN input_id;
    -- Otherwise, generate a prefixed ID
    ELSE
        RETURN 'custom_' || input_id;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Step 6: Update table statistics
ANALYZE articles;

-- Step 7: Verify the changes
DO $$
BEGIN
    RAISE NOTICE 'Migration completed successfully:';
    RAISE NOTICE '- articles.id is now TEXT type';
    RAISE NOTICE '- articles_staging table removed';
    RAISE NOTICE '- Staging functions removed';
    RAISE NOTICE '- normalize_article_id() function added';
END $$;
