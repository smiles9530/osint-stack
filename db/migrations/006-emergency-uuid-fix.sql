-- Emergency UUID Fix for Reddit IDs
\connect osint

-- First, let's see what we're dealing with
SELECT 'Current invalid UUIDs:' as status;
SELECT id, url, LEFT(id::text, 20) as id_preview, created_at 
FROM articles 
WHERE id::text !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
LIMIT 10;

-- Create a function to fix Reddit IDs
CREATE OR REPLACE FUNCTION fix_reddit_ids()
RETURNS INTEGER AS $$
DECLARE
    article_record RECORD;
    new_uuid UUID;
    fixed_count INTEGER := 0;
BEGIN
    -- Loop through articles with invalid UUIDs
    FOR article_record IN 
        SELECT id, url, metadata
        FROM articles 
        WHERE id::text !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    LOOP
        -- Generate new UUID
        new_uuid := gen_random_uuid();
        
        -- Update the article with new UUID
        UPDATE articles 
        SET id = new_uuid,
            metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object(
                'original_invalid_id', article_record.id,
                'fixed_at', NOW(),
                'fix_reason', 'Invalid UUID format'
            )
        WHERE url = article_record.url;
        
        fixed_count := fixed_count + 1;
    END LOOP;
    
    RETURN fixed_count;
END;
$$ LANGUAGE plpgsql;

-- Run the fix
SELECT 'Fixing invalid UUIDs...' as status;
SELECT fix_reddit_ids() as articles_fixed;

-- Verify the fix
SELECT 'Verification - remaining invalid UUIDs:' as status;
SELECT COUNT(*) as remaining_invalid_uuids
FROM articles 
WHERE id::text !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$';

-- Show some fixed articles
SELECT 'Sample of fixed articles:' as status;
SELECT id, url, metadata->'original_invalid_id' as original_id, metadata->'fixed_at' as fixed_at
FROM articles 
WHERE metadata ? 'original_invalid_id'
LIMIT 5;

SELECT 'Emergency UUID fix completed!' as status;
